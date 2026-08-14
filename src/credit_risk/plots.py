"""Regenerate the portfolio figures from the tracked experiment artifacts.

Every figure is built ONLY from the JSON/CSV files in experiments/ (all
committed), so the charts reproduce with no dataset, no model, and no training:

    python -m credit_risk.plots

Figures are written as PNGs to docs/imgs/ and embedded in the README and the
model card. Keeping generation separate from the analysis code means a reader
can rebuild every chart in seconds from the numbers the pipeline already logged.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never needs a display (CI-safe)
import matplotlib.pyplot as plt
import pandas as pd

from credit_risk import config

IMG_DIR = config.DOCS_DIR / "imgs"

# A small, consistent palette so every figure reads as one set.
INK = "#22303c"        # titles / primary text
MUTED = "#6c757d"      # references, secondary marks
GOOD = "#2a9d8f"       # the data-calibrated / chosen option
BAD = "#e76f51"        # the fixed-assumption / weaker option
BAR = "#4c72b0"        # neutral bars
BAR_ALT = "#dd8452"    # second series


def _style() -> None:
    """Apply a clean, presentation-friendly default look (matplotlib only)."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#c9ced3",
        "axes.grid": True,
        "grid.color": "#e6e9ec",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _load_json(name: str) -> dict:
    return json.loads((config.EXPERIMENTS_DIR / name).read_text())


def profit_curve(out: Path) -> None:
    """The money shot: realized profit vs the approve/decline threshold.

    Shows that a cutoff calibrated from the data's own realized cashflows lands
    next to the post-hoc optimum, while a fixed-assumption cutoff leaves roughly
    half the profit on the table.
    """
    curve = pd.read_csv(config.EXPERIMENTS_DIR / "profit_curve.csv")
    study = _load_json("threshold_study.json")

    m = 1e6  # plot dollars in millions
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve["threshold"], curve["realized_profit"] / m,
            color=INK, lw=2, zorder=2)
    ax.axhline(0, color=MUTED, lw=1, ls=":")

    points = [
        ("oracle", "Oracle (post-hoc max)", MUTED, "s"),
        ("empirical", "Data-calibrated cutoff", GOOD, "o"),
        ("fixed", "Fixed-assumption cutoff", BAD, "o"),
    ]
    for key, label, color, marker in points:
        t = study[key]["threshold"]
        p = study[key]["realized_profit"] / m
        ax.scatter([t], [p], s=130, color=color, marker=marker,
                   zorder=5, edgecolor="white", linewidth=1.5,
                   label=f"{label}  (${p:,.1f}M)")

    # Focus the view on the decision-relevant region. Approving everyone
    # (threshold -> 1.0) crashes to -$274M; note it rather than let it flatten
    # the informative hump.
    top = curve["realized_profit"].max() / m
    ax.set_ylim(-30, top * 1.18)
    ax.annotate(
        "approve-all cutoff plunges to -$274M (off-chart)",
        xy=(0.985, -28), ha="right", va="bottom", fontsize=9,
        color=BAD, style="italic",
    )

    fixed_p = study["fixed"]["realized_profit"]
    emp_p = study["empirical"]["realized_profit"]
    oracle_p = study["oracle"]["realized_profit"]
    ax.set_title("Set the threshold from data, not from a guess", pad=12)
    # Headline takeaway placed in the empty center-right region (the curve has
    # already plunged off-chart there, so nothing is occluded).
    ax.text(
        0.60, 0.80,
        f"Data-calibrated earns {emp_p / fixed_p:.1f}x the fixed cutoff\n"
        f"and captures {emp_p / oracle_p:.0%} of the oracle profit",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10, color=INK,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f3f5f7",
                  edgecolor="#c9ced3", linewidth=0.8),
    )
    ax.set_xlabel("Decision threshold (approve if P(default) <= t)")
    ax.set_ylabel("Realized profit on the test book (millions USD)")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower left", framealpha=0.95, edgecolor="#c9ced3")
    fig.savefig(out)
    plt.close(fig)


def shap_importance(out: Path) -> None:
    """Mean absolute SHAP value per feature (global driver ranking)."""
    imp = _load_json("shap_importance.json")
    items = sorted(imp.items(), key=lambda kv: kv[1])  # ascending for barh
    names = [k for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = [GOOD if i >= len(vals) - 6 else BAR for i in range(len(vals))]
    ax.barh(names, vals, color=colors)
    ax.set_title("What drives the prediction (mean |SHAP|)")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, y, f"{v:.02f}", va="center",
                fontsize=8.5, color=MUTED)
    ax.margins(x=0.12)
    fig.savefig(out)
    plt.close(fig)


def temporal_stability(out: Path) -> None:
    """Out-of-time ROC-AUC by loan vintage: is the model stable over years?"""
    val = _load_json("validation.json")
    pooled = _load_json("metrics.json")["roc_auc"]
    rows = val["temporal"]
    years = [str(r["test_year"]) for r in rows]
    aucs = [r["roc_auc"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(years, aucs, color=BAR, width=0.68)
    ax.axhline(pooled, color=GOOD, lw=2, ls="--",
               label=f"pooled ROC-AUC = {pooled:.3f}")
    ax.set_ylim(0.5, 0.78)
    ax.set_title("Out-of-time ROC-AUC by vintage (no drift)")
    ax.set_xlabel("Test year")
    ax.set_ylabel("ROC-AUC")
    ax.grid(axis="x", visible=False)
    for x, v in enumerate(aucs):
        ax.text(x, v + 0.006, f"{v:.02f}", ha="center", fontsize=8.5,
                color=MUTED)
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.text(len(years) - 0.5, 0.507, "random (0.50)", ha="right",
            fontsize=8.5, color=MUTED, style="italic")
    ax.legend(loc="lower right", framealpha=0.95, edgecolor="#c9ced3")
    fig.savefig(out)
    plt.close(fig)


def model_and_calibration(out: Path) -> None:
    """Two panels: GBM vs a logistic scorecard, and the calibration gain."""
    val = _load_json("validation.json")
    cmp = val["comparison"]
    cal = val["calibration"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.6))

    # Panel A: model comparison on ranking metrics.
    metrics = ["ROC-AUC", "PR-AUC"]
    lgbm = [cmp["lightgbm"]["roc_auc"], cmp["lightgbm"]["pr_auc"]]
    logit = [cmp["logistic"]["roc_auc"], cmp["logistic"]["pr_auc"]]
    x = range(len(metrics))
    w = 0.36
    a.bar([i - w / 2 for i in x], lgbm, width=w, color=BAR, label="LightGBM")
    a.bar([i + w / 2 for i in x], logit, width=w, color=BAR_ALT,
          label="Logistic scorecard")
    a.set_xticks(list(x))
    a.set_xticklabels(metrics)
    a.set_ylim(0, 0.8)
    a.set_title("GBM vs an interpretable baseline")
    a.set_ylabel("Score")
    a.grid(axis="x", visible=False)
    for i, (g, lo) in enumerate(zip(lgbm, logit, strict=True)):
        a.text(i - w / 2, g + 0.012, f"{g:.03f}", ha="center", fontsize=8.5,
               color=MUTED)
        a.text(i + w / 2, lo + 0.012, f"{lo:.03f}", ha="center", fontsize=8.5,
               color=MUTED)
    a.legend(loc="lower center", fontsize=9, framealpha=0.95,
             edgecolor="#c9ced3")

    # Panel B: Brier score before/after isotonic calibration (lower is better).
    labels = ["Uncalibrated", "Isotonic"]
    briers = [cal["brier_uncalibrated"], cal["brier_calibrated"]]
    b.bar(labels, briers, color=[BAD, GOOD], width=0.6)
    b.set_ylim(0, max(briers) * 1.25)
    b.set_title("Calibration cuts the Brier score")
    b.set_ylabel("Brier score (lower is better)")
    b.grid(axis="x", visible=False)
    for i, v in enumerate(briers):
        b.text(i, v + max(briers) * 0.02, f"{v:.03f}", ha="center",
               fontsize=9.5, color=INK)
    fig.savefig(out)
    plt.close(fig)


def ablation(out: Path) -> None:
    """How much ranking power does LC's own risk pricing add on top of ours?"""
    ab = _load_json("ablation.json")
    labels = ["Independent model\n(borrower + bureau)",
              "+ LC grade / sub_grade / int_rate"]
    aucs = [ab["primary"]["roc_auc"], ab["with_lc_risk"]["roc_auc"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, aucs, color=[GOOD, MUTED], width=0.58)
    ax.set_ylim(0.5, 0.80)
    ax.set_title("LC's proprietary score adds little ranking power")
    ax.set_ylabel("ROC-AUC")
    ax.grid(axis="x", visible=False)
    for i, v in enumerate(aucs):
        ax.text(i, v + 0.006, f"{v:.03f}", ha="center", fontsize=11,
                color=INK, fontweight="bold")
    lift = ab["roc_auc_lift"]
    ax.text(
        0.03, 0.96,
        f"+{lift:.3f} ROC-AUC from LC's own risk pricing.\n"
        "~98% of the ranking power comes from\n"
        "transparent borrower + bureau features.",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10, color=INK,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f3f5f7",
                  edgecolor="#c9ced3", linewidth=0.8),
    )
    fig.savefig(out)
    plt.close(fig)


FIGURES = {
    "profit_curve.png": profit_curve,
    "shap_importance.png": shap_importance,
    "temporal_stability.png": temporal_stability,
    "model_comparison.png": model_and_calibration,
    "ablation.png": ablation,
}


def main() -> None:
    _style()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for filename, fn in FIGURES.items():
        path = IMG_DIR / filename
        fn(path)
        print(f"saved -> {path.relative_to(config.REPO_ROOT)}")


if __name__ == "__main__":
    main()
