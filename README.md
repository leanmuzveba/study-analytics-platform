# StudyTok Analytics Platform

Cloud-hosted platform that analyzes student study behavior (study hours,
active recall scores, rest intervals) and predicts exam performance,
built for Lean Muzveba's portfolio.

## Components

| Folder         | Purpose                                                              |
|----------------|-----------------------------------------------------------------------|
| `cpp_engine/`  | Standalone C++ engine — parses/filters raw CSV study logs before handoff |
| `backend/`     | Python (FastAPI) API — data pipeline, ML inference, AWS integration  |
| `ml/`          | Model training notebooks + saved regression models                   |
| `infra/`       | AWS deployment scripts (Elastic Beanstalk/Lambda, S3, RDS)           |
| `data/`        | `raw/` (unprocessed CSV logs), `processed/` (cleaned datasets)       |
| `frontend/`    | Web dashboard                                                        |

## Build order

1. ✅ C++ CSV parsing engine
2. ✅ Python data pipeline (pandas cleaning + features)
3. ✅ ML regression model (scikit-learn)
4. ✅ Backend API wrapping the model
5. AWS infrastructure
6. Frontend dashboard
7. Deploy + wire together

## Running the backend locally

```bash
# 1. Build the C++ engine (backend calls it as a subprocess)
g++ -std=c++17 -O2 -Icpp_engine/include \
    cpp_engine/src/main.cpp cpp_engine/src/csv_parser.cpp cpp_engine/src/validator.cpp \
    -o cpp_engine/build/studytok_engine

# 2. Train the model (needed before /api/predict or /api/logs/upload will work)
cd ml
pip install -r ../backend/requirements.txt
python generate_synthetic_data.py ../data/raw/synthetic_logs.csv 220
../cpp_engine/build/studytok_engine ../data/raw/synthetic_logs.csv ../data/processed/synthetic_clean.csv
python feature_engineering.py ../data/processed/synthetic_clean.csv ../data/processed/synthetic_features.csv
python train_model.py ../data/processed/synthetic_features.csv models/exam_score_predictor.joblib

# 3. Run the API
cd ../backend
uvicorn app.main:app --reload
# Interactive docs at http://127.0.0.1:8000/docs
```

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/model/info` | Model type, feature list, test metrics |
| POST | `/api/predict` | Predict exam score from a feature vector |
| POST | `/api/logs/upload` | Upload raw CSV → validate (C++) → aggregate features → predict for unlabeled students |

## Status

Backend API functional end-to-end: upload a raw study log CSV, it gets validated
by the C++ engine, aggregated into per-student features, and predicted on by the
trained regression model — all through one HTTP request.
