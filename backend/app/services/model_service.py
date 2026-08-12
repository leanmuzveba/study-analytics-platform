"""
Loads the trained exam score regression model (produced by
ml/train_model.py) and exposes prediction functions to the API layer.

Model artifacts live in ml/models/ at the repo root — this module
resolves that path relative to this file so it works regardless of
where the app is launched from.
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

# repo_root/backend/app/services/model_service.py -> repo_root
REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = REPO_ROOT / "ml" / "models" / "exam_score_predictor.joblib"
META_PATH = REPO_ROOT / "ml" / "models" / "exam_score_predictor.meta.json"


class ModelNotTrainedError(RuntimeError):
    """Raised when prediction is requested but no trained model exists yet."""


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise ModelNotTrainedError(
            f"No trained model found at {MODEL_PATH}. "
            "Run ml/train_model.py first."
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    if not META_PATH.exists():
        raise ModelNotTrainedError(
            f"No model metadata found at {META_PATH}. "
            "Run ml/train_model.py first."
        )
    with open(META_PATH) as f:
        return json.load(f)


def predict(features: dict) -> float:
    """
    features: dict with exactly the keys in metadata['feature_columns'].
    Returns the predicted exam score as a float.
    """
    model = _load_model()
    meta = load_metadata()
    columns = meta["feature_columns"]

    row = pd.DataFrame([{col: features[col] for col in columns}])
    prediction = model.predict(row)[0]
    return float(prediction)


def predict_batch(features_df: pd.DataFrame) -> list[float]:
    """features_df must contain all of metadata['feature_columns']."""
    model = _load_model()
    meta = load_metadata()
    columns = meta["feature_columns"]
    return [float(p) for p in model.predict(features_df[columns])]
