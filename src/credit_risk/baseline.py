"""Logistic-regression scorecard baseline.

The industry-standard credit model is a logistic-regression scorecard: interpretable and regulator-friendly.
It is the reference the gradient-boosted model must beat to justify the extra complexity.
Numeric features are imputed and scaled, categoricals are one-hot encoded, all inside one sklearn Pipeline so the
same object trains and serves.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from credit_risk import config


def _categoricals_to_object(X: pd.DataFrame) -> pd.DataFrame:
    """Cast 'category' columns to object so the imputer/encoder handle them.
    Module-level (not a lambda) so the fitted pipeline stays picklable."""
    X = X.copy()
    for col in X.select_dtypes(include="category").columns:
        X[col] = X[col].astype(object)
    return X


def build_logistic() -> Pipeline:
    """An unfitted logistic-regression pipeline over the standard feature set."""
    numeric = [c for c in config.FEATURE_COLUMNS if c not in config.CATEGORICAL_FEATURES]
    categorical = list(config.CATEGORICAL_FEATURES)
    preprocess = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("to_object", FunctionTransformer(_categoricals_to_object)),
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def train_logistic(X: pd.DataFrame, y) -> Pipeline:
    """Fit the logistic-regression baseline on the engineered feature matrix."""
    pipe = build_logistic()
    pipe.fit(X, y)
    return pipe
