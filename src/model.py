"""Model training, evaluation, and persistence utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler


TIER_BOUNDARIES = (10, 25)
TIER_LABELS = ("Low", "Medium", "High")


def assign_demand_tier(qty: np.ndarray) -> np.ndarray:
    low, high = TIER_BOUNDARIES
    tier = np.where(qty < low, 0, np.where(qty < high, 1, 2))
    return tier.astype(int)


def tier_label(idx: int) -> str:
    return TIER_LABELS[idx]


@dataclass
class RegressionResult:
    mae: float
    rmse: float
    r2: float
    cv_mae_mean: float
    cv_mae_std: float


@dataclass
class ClassificationResult:
    accuracy: float
    f1_macro: float
    f1_per_class: dict
    confusion: list
    report: dict


def evaluate_regression(model, X_test, y_test, cv_mae_mean, cv_mae_std) -> RegressionResult:
    preds = np.clip(model.predict(X_test), 1.0, None)
    return RegressionResult(
        mae=float(mean_absolute_error(y_test, preds)),
        rmse=float(np.sqrt(mean_squared_error(y_test, preds))),
        r2=float(r2_score(y_test, preds)),
        cv_mae_mean=float(cv_mae_mean),
        cv_mae_std=float(cv_mae_std),
    )


def evaluate_classification(model, X_test, y_test) -> ClassificationResult:
    preds = model.predict(X_test)
    f1_each = f1_score(y_test, preds, average=None, labels=[0, 1, 2], zero_division=0)
    return ClassificationResult(
        accuracy=float(accuracy_score(y_test, preds)),
        f1_macro=float(f1_score(y_test, preds, average="macro", zero_division=0)),
        f1_per_class={TIER_LABELS[i]: float(f1_each[i]) for i in range(3)},
        confusion=confusion_matrix(y_test, preds, labels=[0, 1, 2]).tolist(),
        report=classification_report(y_test, preds, labels=[0, 1, 2],
                                      target_names=list(TIER_LABELS),
                                      output_dict=True, zero_division=0),
    )


def cross_validate_mae(model, X, y, n_splits=5, random_state=42):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = -cross_val_score(model, X, y, cv=cv,
                              scoring="neg_mean_absolute_error", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def build_regressor(n_features: int, random_state: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=3,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
    )


def build_classifier(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )


def save_artifacts(output_dir: Path, regressor, classifier, scaler,
                    pca, feature_cols: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(regressor, output_dir / "regressor.pkl")
    joblib.dump(classifier, output_dir / "classifier.pkl")
    joblib.dump(scaler, output_dir / "scaler.pkl")
    joblib.dump(pca, output_dir / "pca.pkl")
    pd.Series(feature_cols).to_csv(output_dir / "feature_columns.csv",
                                     index=False, header=False)
