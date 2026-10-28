import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.train import build_pipeline, reference_profile


def sample_customer(**changes):
    customer = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 12,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "No",
        "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 55.0, "TotalCharges": 660.0,
    }
    customer.update(changes)
    return customer


@pytest.fixture
def customer():
    return sample_customer()


@pytest.fixture
def client(tmp_path, monkeypatch):
    rows = [sample_customer(), sample_customer(gender="Male", tenure=60, Contract="Two year", MonthlyCharges=25.0, TotalCharges=1500.0), sample_customer(InternetService="Fiber optic", MonthlyCharges=90.0, TotalCharges=1080.0), sample_customer(Partner="No", Dependents="Yes", tenure=36, TotalCharges=1980.0)]
    frame = pd.DataFrame(rows)
    pipeline = build_pipeline(frame)
    pipeline.fit(frame, [1, 0, 1, 0])
    artifact = tmp_path / "model.joblib"
    joblib.dump({"pipeline": pipeline, "threshold": 0.4, "metadata": {"trained_at": "2026-01-01T00:00:00Z", "reference_profile": reference_profile(frame)}}, artifact)
    monkeypatch.setenv("MODEL_PATH", str(artifact))
    monkeypatch.setenv("CHURN_API_KEY", "test-key")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def headers():
    return {"X-API-Key": "test-key"}
