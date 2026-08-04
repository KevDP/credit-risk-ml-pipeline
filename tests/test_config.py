"""Tests for the target definition in credit_risk.config.

Pure-logic tests: no dataset, no trained model, so they run in milliseconds and guard the single most important business rule
in the project (what counts as a default).
"""
from __future__ import annotations

from credit_risk.config import resolve_label


def test_charged_off_is_default() -> None:
    assert resolve_label("Charged Off") == 1


def test_fully_paid_is_not_default() -> None:
    assert resolve_label("Fully Paid") == 0


def test_in_progress_loan_has_no_label() -> None:
    # A current loan has no known outcome and must be excluded, not labeled 0.
    assert resolve_label("Current") is None


def test_credit_policy_variants_are_mapped() -> None:
    assert resolve_label("Does not meet the credit policy. Status:Charged Off") == 1
    assert resolve_label("Does not meet the credit policy. Status:Fully Paid") == 0
