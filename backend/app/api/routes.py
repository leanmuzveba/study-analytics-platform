"""API routes for the StudyTok Analytics platform."""

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.prediction import (
    StudentFeatures, PredictionResponse, UploadSummary,
    StudentPrediction, ModelInfo,
)
from app.services import model_service, csv_engine_service

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/model/info", response_model=ModelInfo)
def model_info():
    try:
        return model_service.load_metadata()
    except model_service.ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/predict", response_model=PredictionResponse)
def predict(features: StudentFeatures, student_id: str | None = None):
    """
    Predict exam score from an already-computed feature vector.
    Useful once the frontend has pulled a student's aggregated stats
    and just needs a prediction.
    """
    try:
        score = model_service.predict(features.model_dump())
    except model_service.ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return PredictionResponse(student_id=student_id, predicted_exam_score=score)


@router.post("/logs/upload", response_model=UploadSummary)
async def upload_logs(file: UploadFile = File(...), predict_unlabeled: bool = True):
    """
    Upload a raw study-log CSV. Runs it through the C++ validation
    engine, aggregates it into per-student features, and — by default —
    predicts exam scores for any students who don't have one yet.
    """
    raw_bytes = await file.read()

    try:
        summary, features_df = csv_engine_service.process_raw_csv(raw_bytes)
    except csv_engine_service.CppEngineNotBuiltError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {e}")

    labeled = features_df[features_df["target_exam_score"].notna()]
    unlabeled = features_df[features_df["target_exam_score"].isna()]

    predictions: list[StudentPrediction] = []
    if predict_unlabeled and len(unlabeled) > 0:
        try:
            preds = model_service.predict_batch(unlabeled)
            predictions = [
                StudentPrediction(student_id=sid, predicted_exam_score=score)
                for sid, score in zip(unlabeled["student_id"], preds)
            ]
        except model_service.ModelNotTrainedError:
            pass  # no model yet — still return the parse/feature summary

    return UploadSummary(
        rows_read=summary.rows_read,
        rows_accepted=summary.rows_accepted,
        rows_rejected=summary.rows_rejected,
        students_processed=len(features_df),
        students_with_labels=len(labeled),
        students_awaiting_prediction=len(unlabeled),
        predictions=predictions,
    )
