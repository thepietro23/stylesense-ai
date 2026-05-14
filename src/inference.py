"""Inference layer.

Loads trained artefacts (ResNet50, PCA, scaler, regressor, classifier, metrics)
and exposes a single function that maps an input image and price to a predicted
quantity, a confidence range, and a demand tier.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from src.feature_extractor import extract_embeddings, load_resnet50_extractor
from src.model import TIER_LABELS


@dataclass
class Prediction:
    point_estimate: float
    lower: float
    upper: float
    confidence_width: float
    tier: str
    tier_probabilities: dict


@lru_cache(maxsize=1)
def _load_pipeline(model_dir: str, processed_dir: str) -> dict:
    model_dir_p = Path(model_dir)
    processed_dir_p = Path(processed_dir)

    regressor = joblib.load(model_dir_p / "regressor.pkl")
    classifier = joblib.load(model_dir_p / "classifier.pkl")
    scaler = joblib.load(model_dir_p / "scaler.pkl")
    pca = joblib.load(model_dir_p / "pca.pkl")
    feature_cols = pd.read_csv(model_dir_p / "feature_columns.csv",
                                header=None)[0].tolist()
    metrics = json.loads((model_dir_p / "metrics.json").read_text())

    extractor = load_resnet50_extractor(device="cpu")

    return {
        "regressor": regressor,
        "classifier": classifier,
        "scaler": scaler,
        "pca": pca,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "extractor": extractor,
    }


def predict_quantity(
    image_path: str | Path,
    rate: float,
    model_dir: str | Path = "models",
    processed_dir: str | Path = "data/processed",
) -> Prediction:
    """Predict expected sales quantity, range, and demand tier for an image.

    The regressor and classifier predictions are smoothed across a small price
    window centred on the user's rate (rate ± 150 in five steps). This averages
    out spurious local non-monotonicities in the tree-based models without
    materially changing aggregate metrics.
    """
    pipeline = _load_pipeline(str(model_dir), str(processed_dir))

    with Image.open(image_path) as im:
        im.convert("RGB")
    embedding = extract_embeddings([image_path], model=pipeline["extractor"],
                                     device="cpu", batch_size=1)[0]

    pca_features = pipeline["pca"].transform(embedding.reshape(1, -1))[0]
    rate_window = np.linspace(max(400.0, rate - 150), min(1700.0, rate + 150), 5)
    batch = np.array([np.hstack([pca_features, [r]]) for r in rate_window])
    scaled_batch = pipeline["scaler"].transform(batch)

    qty_predictions = pipeline["regressor"].predict(scaled_batch)
    probs_batch = pipeline["classifier"].predict_proba(scaled_batch)

    point = float(np.mean(qty_predictions))
    point = max(point, 1.0)

    avg_probs = probs_batch.mean(axis=0)
    tier_idx = int(np.argmax(avg_probs))
    tier_name = TIER_LABELS[tier_idx]
    tier_probs = {TIER_LABELS[i]: round(float(avg_probs[i]), 3)
                   for i in range(len(avg_probs))}

    rmse = pipeline["metrics"]["regressor"]["test_rmse"]
    lower = max(point - rmse, 1.0)
    upper = point + rmse

    return Prediction(
        point_estimate=round(point, 1),
        lower=round(lower, 1),
        upper=round(upper, 1),
        confidence_width=round(rmse, 1),
        tier=tier_name,
        tier_probabilities=tier_probs,
    )
