"""
StudyTok Analytics — exam score regression model.

Trains on the per-student feature table produced by feature_engineering.py.
Compares a simple linear model against a random forest, picks whichever
generalizes better on held-out data, and saves it for the backend to load.

Usage:
    python train_model.py [features_csv] [model_output_path]

Defaults:
    features_csv: ../data/processed/student_features.csv
    model_output: models/exam_score_predictor.joblib
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURE_COLUMNS = [
    "num_logs", "avg_study_hours", "std_study_hours", "study_hours_trend",
    "avg_active_recall_score", "recall_score_trend", "avg_rest_hours",
    "rest_deficit_rate", "avg_sessions_count", "avg_distraction_events",
]
TARGET_COLUMN = "target_exam_score"


def evaluate(model, X_test, y_test, name: str) -> dict:
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds)
    print(f"  {name:<20} MAE={mae:6.2f}  RMSE={rmse:6.2f}  R2={r2:.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    features_path = sys.argv[1] if len(sys.argv) > 1 else "../data/processed/student_features.csv"
    model_out = sys.argv[2] if len(sys.argv) > 2 else "models/exam_score_predictor.joblib"

    df = pd.read_csv(features_path)

    labeled = df[df[TARGET_COLUMN].notna()].copy()
    unlabeled = df[df[TARGET_COLUMN].isna()].copy()

    print("StudyTok exam score model — training")
    print(f"  Features file:     {features_path}")
    print(f"  Labeled students:  {len(labeled)} (usable for training)")
    print(f"  Unlabeled students:{len(unlabeled):>4} (awaiting prediction)")

    if len(labeled) < 20:
        print("\n  Not enough labeled students to train/test split meaningfully "
              "(need at least ~20). Run generate_synthetic_data.py for a larger "
              "dataset, or wait for more real student logs.")
        return

    X = labeled[FEATURE_COLUMNS]
    y = labeled[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n  Train/test split: {len(X_train)} / {len(X_test)}")
    print("\n  Model comparison (held-out test set):")

    # Linear regression — interpretable baseline
    linear_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])
    linear_pipeline.fit(X_train, y_train)
    linear_metrics = evaluate(linear_pipeline, X_test, y_test, "Linear regression")

    # Random forest — captures non-linear interactions
    forest = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    forest.fit(X_train, y_train)
    forest_metrics = evaluate(forest, X_test, y_test, "Random forest")

    # Pick the better model by test RMSE
    if forest_metrics["rmse"] < linear_metrics["rmse"]:
        best_model, best_name, best_metrics = forest, "random_forest", forest_metrics
    else:
        best_model, best_name, best_metrics = linear_pipeline, "linear_regression", linear_metrics

    print(f"\n  Selected model: {best_name} (lower test RMSE)")

    # What drives the prediction? (helps power "suggest optimized study
    # schedules" later — the features with the most influence)
    if best_name == "random_forest":
        importances = sorted(
            zip(FEATURE_COLUMNS, best_model.feature_importances_),
            key=lambda p: p[1], reverse=True
        )
        print("\n  Top feature importances:")
        for feat, imp in importances[:5]:
            print(f"    {feat:<25} {imp:.3f}")
    else:
        coefs = sorted(
            zip(FEATURE_COLUMNS, best_model.named_steps["model"].coef_),
            key=lambda p: abs(p[1]), reverse=True
        )
        print("\n  Top standardized coefficients:")
        for feat, coef in coefs[:5]:
            print(f"    {feat:<25} {coef:+.3f}")

    # Save the model + metadata needed to use it correctly later
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_out)

    meta_path = str(Path(model_out).with_suffix(".meta.json"))
    with open(meta_path, "w") as f:
        json.dump({
            "model_type": best_name,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "test_metrics": best_metrics,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }, f, indent=2)

    print(f"\n  Saved model:    {model_out}")
    print(f"  Saved metadata: {meta_path}")

    # Demonstrate applying it to students awaiting prediction
    if len(unlabeled) > 0:
        preds = best_model.predict(unlabeled[FEATURE_COLUMNS])
        print(f"\n  Sample predictions for {min(5, len(unlabeled))} unlabeled students:")
        for sid, pred in list(zip(unlabeled["student_id"], preds))[:5]:
            print(f"    {sid}: predicted exam score = {pred:.1f}")


if __name__ == "__main__":
    main()
