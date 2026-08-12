"""
StudyTok Analytics API — entry point.

Run locally:
    cd backend
    uvicorn app.main:app --reload

Endpoints are registered under /api — see app/api/routes.py.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router

app = FastAPI(
    title="StudyTok Analytics API",
    description="Predicts exam performance from student study behavior logs.",
    version="0.1.0",
)

# Wide open for now — tighten this to the actual frontend origin once
# it's deployed (Step 6/7).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.get("/")
def root():
    return {"service": "studytok-analytics-api", "docs": "/docs"}
