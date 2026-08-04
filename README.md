# Credit Risk ML Pipeline

End-to-end, scale-to-zero pipeline that predicts consumer loan default and serves the model behind a serverless API. 
Built as a business case: it starts from a lending decision and its asymmetric costs, not from a leaderboard metric.

> Status: in active development. This README documents the design
> Results are filled in as each phase lands (see Roadmap).

## The business problem

A lender approving a loan faces an asymmetric bet. Approving a borrower who defaults loses most of the principal. 
Declining a borrower who would have repaid loses only the interest margin. A model that optimizes raw accuracy ignores this
asymmetry and ships the wrong decision threshold.

This project treats the problem the way a lender does:

- Predict the probability that a loan defaults.
- Choose the decision threshold from an expected-cost curve, not from 0.5.
- Keep the model honest about what it knew at application time (no leakage).
- Explain individual decisions, because credit decisions must be justifiable.

## Dataset

**Lending Club** (2007 to 2018 Q4): real peer-to-peer consumer loans with borrower, loan, and credit-bureau attributes and the final loan status. The label (`default`) is defined from `loan_status`; loans still in progress are
excluded because their outcome is unknown. See `data/README.md` for download instructions (the data is not committed).

## Approach

The project spans the full lifecycle an ML engineer owns, not just the model.

1. **Data science (notebook).** EDA, a defensible feature set, honest validation (leakage removal, cross-validation), model comparison, and threshold selection by expected business cost.
2. **Engineering (`src/`).** The same logic refactored into a tested, notebook-independent Python package: data, features, training, evaluation, and inference.
3. **Serving (serverless).** The model behind AWS Lambda + API Gateway, scaling to zero so idle cost is effectively $0.

## Architecture

```
Client  -> API Gateway
        -> Lambda (inference) <- model artifact in S3
        -> prediction                  
```

Training runs on demand (locally or as an ephemeral SageMaker training job) and writes the model artifact to S3.
Serving is fully serverless. Nothing runs whenno one is calling it, which is the design constraint: no always-on resources,
teardown with a single `terraform destroy`.

## Repository structure

```
src/credit_risk/    Python package (data, features, train, evaluate, inference)
notebooks/          EDA and modeling narrative
tests/              pytest suite (business rules, features, inference)
serving/            Lambda handler + Dockerfile
infra/              Terraform for the serverless stack
data/               download instructions (data is gitignored)
models/             model artifacts (gitignored)
experiments/        metrics logged per run
docs/               architecture notes and model card
```

## Quickstart

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -e ".[dev,serving]"
pytest -q
```

## Cost

Designed to stay near $0 at rest:

- Serverless inference (Lambda + API Gateway)
- Ephemeral training
- S3 for storage 
- Budget alarm

## Roadmap

- Phase 0: scaffolding
- Phase 1: EDA, feature engineering, validation, threshold selection
- Phase 2: refactor to a tested `src/` package + model card
- Phase 3: serverless serving (Lambda + API Gateway) + Terraform
- Phase 4: README results, architecture diagram, write-up

## Author

Kevin Delgado.
Former AI/Data roles at AWS and Deloitte.
