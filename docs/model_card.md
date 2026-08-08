# Model Card: Credit Default Risk (Lending Club)

A binary classifier that estimates the probability a consumer loan defaults, built as an **independent** risk model from application-time borrower and bureau features. It deliberately excludes Lending Club's own risk-model outputs (grade, sub-grade, interest rate) so that it learns risk from primary signals rather than re-deriving a proprietary score.

## Intended use

- **Use:** rank loan applications by default risk and support an approve/decline decision via a cost-based threshold. Educational / portfolio project.
- **Out of scope:** production credit decisions. Not fair-lending audited and trained on a single lender's historical book.

## Data

- **Source:** Lending Club accepted loans, 2007 to 2018 Q4 (`wordsforthewise/lending-club` on Kaggle). Not committed; see `data/README.md`.
- **Unit:** one issued loan.

## Target

Binary `default` derived from `loan_status`:

- `1` (default): Charged Off, Default, and the "credit policy" charged-off variant.
- `0` (repaid): Fully Paid and its "credit policy" variant.
- **Dropped:** loans still in progress (Current, Late, In Grace Period, Issued).
  Their outcome is unknown, so they cannot be used as supervision.

## Features

**Allowlist, not blocklist.** Only 20 columns known at application time are eligible, so no post-origination field can leak in.

- **Numeric (direct):** loan_amnt, annual_inc, dti, open_acc, pub_rec, revol_bal,
revol_util, total_acc, delinq_2yrs, inq_last_6mths, mort_acc, pub_rec_bankruptcies.
- **Numeric (derived):** term_months, emp_length_years, fico_score (mean of the
  FICO range), credit_history_length (years from earliest credit line to issue).
- **Categorical (native to LightGBM):** home_ownership, verification_status,
  purpose, addr_state.

**Excluded by design:** grade, sub_grade, int_rate (LC's risk-model outputs) and installment (a deterministic function of int_rate).
See the ablation below.

**Leakage guard:** post-origination cashflow, hardship, and settlement columns are enumerated and asserted to be disjoint from the feature set by a test.

## Training and validation

- **Model:** LightGBM, `scale_pos_weight` set to the negative/positive ratio to offset the class imbalance. Categorical columns passed as the native dtype.
- **Split:** time-based. The most recent 20% of loans by issue date are the test set; earlier vintages train. This mirrors deployment (train on the past, score the future) and avoids the optimistic bias of a random split on time-ordered data.

## Performance (held-out future vintages)

| Metric | Value |
|---|---|
| ROC-AUC | 0.703 |
| PR-AUC | 0.384 |
| Test default rate (base rate) | 21.8% |
| Test size | 282,970 |

- PR-AUC of 0.384 against a 0.218 base rate is roughly 1.75x the no-skill baseline.

- ROC-AUC of 0.70 is modest but expected for an independent model that excludes the lender's own risk score.

**Top risk drivers (mean absolute SHAP):** term_months (0.35), fico_score (0.32), loan_amnt (0.18), dti (0.18), annual_inc (0.15), inq_last_6mths (0.13). Loan term and FICO dominate. emp_length and credit_history_length contribute little.

## Baseline comparison

Against a logistic-regression scorecard (the interpretable, regulator-friendly standard in credit), on the same held-out test:

| Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| Logistic scorecard | 0.682 | 0.357 | 0.218 |
| **LightGBM** | **0.703** | **0.384** | **0.215** |

LightGBM improves ROC-AUC by +0.020, enough to justify the added complexity, while the interpretable baseline stays within ~2 points as a fallback.

## Hyperparameter tuning

A 40-trial Optuna search (TPE sampler) over the LightGBM hyperparameters, scored on a time-based validation slice of the training data (the test set left untouched), improved the held-out ROC-AUC by only +0.0009 over the hand-picked defaults. The defaults are kept: the search confirms the hyperparameters are not the performance bottleneck. The ceiling here is set by the features and the intrinsic difficulty of the problem, so feature work, not further tuning, is the lever for future gains.

## Probability calibration

Raw LightGBM scores are miscalibrated (Brier 0.212), in part because `scale_pos_weight` inflates them to offset the class imbalance: class weighting helps ranking but hurts calibration. Isotonic calibration fit on a held-out slice of the training data lowers the test Brier to 0.157 (about 26% better). The served model should return calibrated scores, and the decision threshold should be set on them.

## Temporal stability

Expanding-window backtest: for each vintage year, train on all earlier years and score that year. ROC-AUC stays in a 0.65 to 0.72 band with no drift or collapse.

| Vintage year | ROC-AUC | Loans | Default rate |
|---|---|---|---|
| 2010 | 0.647 | 12,537 | 14.0% |
| 2011 | 0.688 | 21,721 | 15.2% |
| 2012 | 0.668 | 53,367 | 16.2% |
| 2013 | 0.675 | 134,804 | 15.6% |
| 2014 | 0.701 | 223,103 | 18.4% |
| 2015 | 0.722 | 375,546 | 20.2% |
| 2016 | 0.702 | 293,105 | 23.3% |
| 2017 | 0.704 | 169,321 | 23.1% |
| 2018 | 0.711 | 56,318 | 15.8% |

The weakest year (2010) has the smallest training history. From 2014 onward the model is stable around 0.70 to 0.72.

## Ablation: contribution of LC's own risk score

Adding grade, sub_grade, and int_rate back:

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Primary (independent) | 0.703 | 0.384 |
| + LC risk score | 0.715 | 0.395 |
| **Lift** | **+0.012** | **+0.011** |

LC's proprietary score adds only ~1.2 AUC points. Because grade and interest rate are themselves derived from the same bureau signals the model already uses, they carry little independent information. The transparent, independent model captures roughly 98% of the ranking power of the black-box score, which is the reason to exclude it: near-equal accuracy with full interpretability.

## Cost calibration and decision threshold

The approve/decline threshold minimizes expected cost given the asymmetric penalties of approving a defaulter versus declining a good borrower. Costs are **estimated from the data's realized cashflows** (per the IRB / IFRS 9 / CECL practice of deriving loss parameters from history), not assumed:

- **Empirical LGD:** 0.353 of loan_amnt (loss on defaulted loans).
- **Empirical margin:** 0.173 of loan_amnt (profit on repaid loans).

Under a per-dollar cost model the profit-maximizing cutoff is `margin / (LGD + margin)`. A study comparing the data-calibrated cutoff against a plausible fixed assumption (LGD 0.65, margin 0.12), both derived without touching the test set and scored on the test set's realized dollars:

| Policy | Cutoff | Realized profit* | Approval | Book default |
|---|---|---|---|---|
| Fixed assumption | 0.156 | $5.4M | 5.5% | 3.2% |
| **Data-calibrated** | 0.329 | **$11.5M** | 27.2% | 7.9% |
| Oracle (upper bound) | 0.320 | $11.75M | 25.8% | 7.6% |
| Approve all (floor) | 1.000 | -$274.5M | 100% | 21.8% |

The data-calibrated cutoff earns **2.1x** the fixed assumption and captures **98%** of the achievable (oracle) profit, versus 46% for the fixed assumption. The fixed LGD (0.65) was nearly double the empirical (0.35) because borrowers repay part of the loan before defaulting; that single wrong parameter made the fixed cutoff far too conservative.

\* **Absolute dollar magnitudes are distorted by right-censoring** (see
Limitations). The relative comparison is the robust result; the dollar figures are
illustrative, not expected profit.

## Limitations

- **Right-censoring:** the recent test vintages over-represent defaults, because good long-term loans were still "Current" at the data snapshot and were dropped. This inflates the stakes (approve-all looks catastrophic) and distorts absolute dollar figures. A maturity filter would de-bias the magnitudes; the relative findings hold regardless.

- **Threshold on calibrated scores:** isotonic calibration is implemented, but the threshold study was computed on raw scores. Its relative conclusion (data- calibrated costs beat a fixed assumption) is unaffected, since both policies use the same scores; the deployed operating threshold should be re-derived on calibrated scores.
- **Modest discrimination:** ROC-AUC 0.70 reflects both the intrinsic difficulty of the problem and the deliberate exclusion of LC's score.
- **Fair lending:** addr_state and other features can act as proxies for protected classes. A production model would require a disparate-impact audit; this project does not perform one.
- **Single lender, single era:** trained on one platform's 2007 to 2018 book; generalization to other lenders or later periods is untested.

## Reproducing

```bash
pip install -e ".[dev,serving]"
# download data (see data/README.md), then:
python -m credit_risk.train             # metrics.json + model artifact
python -m credit_risk.threshold_study   # threshold_study.json + profit_curve.csv
python -m credit_risk.interpret         # shap_importance.json
python -m credit_risk.ablation          # ablation.json
python -m credit_risk.validate          # validation.json (baseline, calibration, temporal)
python -m credit_risk.tune              # tuning.json (Optuna hyperparameter search)
```

All metrics referenced above are written to `experiments/`.
