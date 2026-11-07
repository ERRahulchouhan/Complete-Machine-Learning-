from __future__ import annotations
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ml_course.pipeline import predict

app = FastAPI(title="Regression inference", version="1.0.0")
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.joblib"))

class PredictionRequest(BaseModel):
    rows: list[dict[str, float | str | None]] = Field(min_length=1, max_length=1000)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "alive"}

@app.get("/ready")
def ready() -> dict[str, str]:
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="Model artifact is unavailable")
    return {"status": "ready"}

@app.post("/predict")
def predict_endpoint(request: PredictionRequest) -> dict[str, list[float]]:
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="Model artifact is unavailable")
    try:
        return {"predictions": predict(MODEL_PATH, request.rows)}
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
