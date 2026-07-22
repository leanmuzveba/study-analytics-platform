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

1. C++ CSV parsing engine
2. Python data pipeline (pandas cleaning + features)
3. ML regression model (scikit-learn)
4. Backend API wrapping the model
5. AWS infrastructure
6. Frontend dashboard
7. Deploy + wire together

## Status

Scaffolding only — components built incrementally, step by step.
