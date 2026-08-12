"""
Tests for the StudyTok Analytics API.

Run from backend/: pytest tests/ -v

Requires the C++ engine to be built and the model to be trained first
(see README) — these tests exercise the real pipeline, not mocks.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_CSV = REPO_ROOT / "data" / "raw" / "synthetic_logs.csv"


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_model_info():
    r = client.get("/api/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_type"] in ("linear_regression", "random_forest")
    assert "avg_study_hours" in body["feature_columns"]
    assert 0 <= body["test_metrics"]["r2"] <= 1


def test_predict_good_habits_scores_higher_than_poor_habits():
    good = {
        "num_logs": 10, "avg_study_hours": 5.0, "std_study_hours": 0.5,
        "study_hours_trend": 0.1, "avg_active_recall_score": 90.0,
        "recall_score_trend": 0.5, "avg_rest_hours": 8.0,
        "rest_deficit_rate": 0.0, "avg_sessions_count": 3.0,
        "avg_distraction_events": 1.0,
    }
    poor = {**good, "avg_study_hours": 1.5, "avg_active_recall_score": 55.0,
            "avg_rest_hours": 4.5, "rest_deficit_rate": 0.8,
            "avg_distraction_events": 6.0}

    r_good = client.post("/api/predict", json=good)
    r_poor = client.post("/api/predict", json=poor)

    assert r_good.status_code == 200
    assert r_poor.status_code == 200
    assert r_good.json()["predicted_exam_score"] > r_poor.json()["predicted_exam_score"]


def test_predict_rejects_invalid_feature_ranges():
    bad = {
        "num_logs": 10, "avg_study_hours": 999.0,  # out of range
        "std_study_hours": 0.5, "study_hours_trend": 0.1,
        "avg_active_recall_score": 90.0, "recall_score_trend": 0.5,
        "avg_rest_hours": 8.0, "rest_deficit_rate": 0.0,
        "avg_sessions_count": 3.0, "avg_distraction_events": 1.0,
    }
    r = client.post("/api/predict", json=bad)
    assert r.status_code == 422  # Pydantic validation error


@pytest.mark.skipif(not SYNTHETIC_CSV.exists(),
                     reason="run ml/generate_synthetic_data.py first")
def test_upload_full_pipeline():
    with open(SYNTHETIC_CSV, "rb") as f:
        r = client.post("/api/logs/upload",
                         files={"file": ("synthetic_logs.csv", f, "text/csv")})

    assert r.status_code == 200
    body = r.json()
    assert body["rows_read"] > 0
    assert body["rows_accepted"] > 0
    assert body["rows_rejected"] > 0  # synthetic data includes bad rows
    assert body["students_processed"] > 0
    assert body["students_awaiting_prediction"] == len(body["predictions"])
