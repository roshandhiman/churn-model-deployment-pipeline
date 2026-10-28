import time

from tests.conftest import sample_customer


def test_health_is_available(client):
    assert client.get("/health").status_code == 200


def test_health_reports_loaded_model(client):
    assert client.get("/health").json()["model_loaded"] is True


def test_predict_requires_key(client, customer):
    assert client.post("/predict", json=customer).status_code == 401


def test_predict_rejects_wrong_key(client, customer):
    assert client.post("/predict", json=customer, headers={"X-API-Key": "wrong"}).status_code == 401


def test_predict_returns_probability(client, headers, customer):
    response = client.post("/predict", json=customer, headers=headers)
    assert response.status_code == 200
    assert 0 <= response.json()["churn_probability"] <= 1


def test_predict_returns_boolean(client, headers, customer):
    assert isinstance(client.post("/predict", json=customer, headers=headers).json()["churn_prediction"], bool)


def test_predict_returns_threshold(client, headers, customer):
    assert client.post("/predict", json=customer, headers=headers).json()["threshold"] == 0.4


def test_batch_returns_all_predictions(client, headers, customer):
    response = client.post("/predict/batch", json={"customers": [customer, sample_customer(gender="Male")]}, headers=headers)
    assert len(response.json()["predictions"]) == 2


def test_empty_batch_is_rejected(client, headers):
    assert client.post("/predict/batch", json={"customers": []}, headers=headers).status_code == 422


def test_negative_tenure_is_rejected(client, headers, customer):
    customer["tenure"] = -1
    assert client.post("/predict", json=customer, headers=headers).status_code == 422


def test_negative_charges_are_rejected(client, headers, customer):
    customer["MonthlyCharges"] = -10
    assert client.post("/predict", json=customer, headers=headers).status_code == 422


def test_zero_tenure_with_charges_is_rejected(client, headers, customer):
    customer["tenure"] = 0
    assert client.post("/predict", json=customer, headers=headers).status_code == 422


def test_missing_field_is_rejected(client, headers, customer):
    customer.pop("gender")
    assert client.post("/predict", json=customer, headers=headers).status_code == 422


def test_unseen_category_is_handled(client, headers, customer):
    customer["InternetService"] = "Satellite"
    assert client.post("/predict", json=customer, headers=headers).status_code == 200


def test_drift_requires_key(client, customer):
    assert client.post("/drift", json={"customers": [customer]}).status_code == 401


def test_drift_returns_report(client, headers, customer):
    response = client.post("/drift", json={"customers": [customer]}, headers=headers)
    assert response.status_code == 200
    assert "psi" in response.json()


def test_drift_flags_shift(client, headers, customer):
    shifted = [sample_customer(MonthlyCharges=900.0, TotalCharges=9000.0, InternetService="NewService") for _ in range(8)]
    assert client.post("/drift", json={"customers": shifted}, headers=headers).json()["drift_detected"] is True


def test_single_prediction_latency(client, headers, customer):
    started = time.perf_counter()
    response = client.post("/predict", json=customer, headers=headers)
    assert response.status_code == 200
    assert time.perf_counter() - started < 0.1
