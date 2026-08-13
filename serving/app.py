"""FastAPI serving app for the credit-risk model.

Exposes POST /predict, which validates a loan application against a Pydantic
schema and returns the default probability plus an approve/decline decision. The
same app runs locally under uvicorn (for testing) and on AWS Lambda via the
Mangum adapter in handler.py, so training, local serving, and production share one
code path through credit_risk.inference.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from credit_risk import inference

app = FastAPI(title="Credit Risk Scoring API", version="0.1.0")


class LoanApplication(BaseModel):
    """Raw application-time fields the model scores. Field names match the dataset
    columns so the record maps straight into feature engineering."""

    loan_amnt: float
    annual_inc: float
    dti: float
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float
    total_acc: float
    delinq_2yrs: float
    inq_last_6mths: float
    mort_acc: float
    pub_rec_bankruptcies: float
    fico_range_low: float
    fico_range_high: float
    term: str = Field(examples=[" 36 months"])
    emp_length: str = Field(examples=["5 years"])
    issue_d: str = Field(examples=["Jan-2018"])
    earliest_cr_line: str = Field(examples=["Jan-2005"])
    home_ownership: str = Field(examples=["RENT"])
    verification_status: str = Field(examples=["Verified"])
    purpose: str = Field(examples=["credit_card"])
    addr_state: str = Field(examples=["CA"])


class ScoreResponse(BaseModel):
    default_probability: float
    decision: str
    threshold: float


def get_model():
    """Model provider. Overridable in tests via app.dependency_overrides."""
    return inference.load_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=ScoreResponse)
def predict(application: LoanApplication, model: Annotated[Any, Depends(get_model)]) -> dict:
    return inference.score(application.model_dump(), model=model)
