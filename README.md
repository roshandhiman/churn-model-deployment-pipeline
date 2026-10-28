# Churn Model Deployment Pipeline

## Overview

This project packages a telecom churn model as a secure FastAPI service. It downloads the public IBM Telco Customer Churn dataset, cleans numeric charges stored as text, encodes categorical service fields, scales numeric features, tunes a class-balanced logistic regression pipeline, and versions the trained artifact.

The API provides a health check, single prediction, batch prediction, API-key authentication, JSON request logs with latency, validation errors, unseen-category handling, and a population stability index drift endpoint.

## Structure

```text
src/data.py       download, cleaning, data hashing
src/train.py      training, tuning, threshold, artifact metadata
src/schemas.py    Pydantic input and output validation
src/main.py       FastAPI service and authentication
src/drift.py      PSI drift calculation
tests/            unit and integration tests
MODEL_CARD.md     model documentation
openapi.yaml      OpenAPI specification
Dockerfile        multi-stage container image
```

## Install and train

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export CHURN_API_KEY=replace-with-a-long-random-secret
python -m src.train
```

Training downloads data on the first run and writes `artifacts/churn_model.joblib` and `artifacts/churn_model.metadata.json`. Metadata contains the training-data SHA-256 hash, metrics, threshold, tuning result, package versions, timestamp, and drift reference profile.

## Run

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for interactive API documentation and `http://localhost:8000/openapi.json` for the generated OpenAPI specification.

## Example request

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -H "X-API-Key: $CHURN_API_KEY" -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,"PhoneService":"Yes","MultipleLines":"No","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":55.0,"TotalCharges":660.0}'
```

## Tests

```bash
pytest -q
```

The suite contains more than 15 checks covering health, API keys, prediction output, batching, validation, unseen categories, drift, and latency.

## Docker

```bash
python -m src.train
docker build -t churn-api .
docker run -p 8000:8000 -e CHURN_API_KEY=replace-with-a-long-random-secret churn-api
```

The multi-stage slim image runs as a non-root user. Train the model before building so the artifact is included.

## Threshold decision

The model does not use a fixed 0.5 threshold. It evaluates thresholds from 0.05 to 0.95 using this cost function:

```text
cost = false_negatives × 500 + false_positives × 100
```

The chosen threshold has minimum validation cost. Missing a real churner is five times more costly than sending a retention offer to a non-churner, so this approach prioritizes recall.

## Safe retraining

Use a versioned data snapshot, train in staging, compare metrics and expected business cost with the active artifact, run tests, register metadata, canary deploy, monitor drift and latency, and retain the previous artifact for rollback.
