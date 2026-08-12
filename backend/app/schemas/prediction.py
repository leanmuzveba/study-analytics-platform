"""Request/response schemas for the StudyTok Analytics API."""

from typing import Optional
from pydantic import BaseModel, Field


class StudentFeatures(BaseModel):
    """
    The per-student feature vector the model expects — matches the
    columns produced by ml/feature_engineering.py exactly.
    """
    num_logs: int = Field(..., ge=1, description="Number of study logs recorded")
    avg_study_hours: float = Field(..., ge=0, le=24)
    std_study_hours: float = Field(..., ge=0)
    study_hours_trend: float
    avg_active_recall_score: float = Field(..., ge=0, le=100)
    recall_score_trend: float
    avg_rest_hours: float = Field(..., ge=0, le=24)
    rest_deficit_rate: float = Field(..., ge=0, le=1)
    avg_sessions_count: float = Field(..., ge=0)
    avg_distraction_events: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    student_id: Optional[str] = None
    predicted_exam_score: float


class StudentPrediction(BaseModel):
    student_id: str
    predicted_exam_score: float


class UploadSummary(BaseModel):
    rows_read: int
    rows_accepted: int
    rows_rejected: int
    students_processed: int
    students_with_labels: int
    students_awaiting_prediction: int
    predictions: list[StudentPrediction] = []


class ModelInfo(BaseModel):
    model_type: str
    feature_columns: list[str]
    target_column: str
    test_metrics: dict
    n_train: int
    n_test: int
