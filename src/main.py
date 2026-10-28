from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from src.drift import drift_report
from src.schemas import BatchRequest, BatchResponse, Customer, DriftRequest, Prediction

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("churn_api")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", "artifacts/churn_model.joblib"))


def require_api_key(api_key: str | None = Security(api_key_header)) -> None:
    expected = os.getenv("CHURN_API_KEY", "change-me")
    if not api_key or api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = model_path()
    app.state.bundle = joblib.load(path) if path.exists() else None
    yield


app = FastAPI(title="Churn Model API", version="1.0.0", description="Secure churn prediction service", lifespan=lifespan)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    logger.info(json.dumps({"method": request.method, "path": request.url.path, "status": response.status_code, "latency_ms": round((time.perf_counter() - started) * 1000, 2)}))
    return response


def get_bundle(request: Request) -> dict:
    if not request.app.state.bundle:
        raise HTTPException(status_code=503, detail="Model artifact is not available")
    return request.app.state.bundle


def predict(records: list[Customer], bundle: dict) -> list[Prediction]:
    frame = pd.DataFrame([record.model_dump() for record in records])
    probabilities = bundle["pipeline"].predict_proba(frame)[:, 1]
    threshold = float(bundle["threshold"])
    return [Prediction(churn_probability=round(float(probability), 6), churn_prediction=bool(probability >= threshold), threshold=threshold) for probability in probabilities]


@app.get("/health")
def health(request: Request):
    bundle = request.app.state.bundle
    return {"status": "ok" if bundle else "degraded", "model_loaded": bool(bundle), "model_version": bundle["metadata"]["trained_at"] if bundle else None}


@app.post("/predict", response_model=Prediction, dependencies=[Depends(require_api_key)])
def single_prediction(request: Request, customer: Customer):
    return predict([customer], get_bundle(request))[0]


@app.post("/predict/batch", response_model=BatchResponse, dependencies=[Depends(require_api_key)])
def batch_prediction(request: Request, payload: BatchRequest):
    return BatchResponse(predictions=predict(payload.customers, get_bundle(request)))


@app.post("/drift", dependencies=[Depends(require_api_key)])
def drift(request: Request, payload: DriftRequest):
    bundle = get_bundle(request)
    records = [customer.model_dump() for customer in payload.customers]
    return drift_report(records, bundle["metadata"]["reference_profile"])
