
import pytest

from inference.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as test_client:
        yield test_client


def valid_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 65.0,
        "TotalCharges": 780.0,
    }


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["model_version"] == "churn-logreg-v1"
    assert data["model_loaded"] is True


def test_valid_prediction(client):
    response = client.post("/predict", json=valid_customer())

    assert response.status_code == 200

    data = response.get_json()
    assert data["model_version"] == "churn-logreg-v1"
    assert data["prediction"] in ["Churn", "No churn"]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["threshold"] == 0.5


def test_missing_features_are_rejected(client):
    response = client.post("/predict", json={})

    assert response.status_code == 400

    data = response.get_json()
    assert "missing_features" in data
    assert "gender" in data["missing_features"]


def test_unexpected_feature_is_rejected(client):
    customer = valid_customer()
    customer["customerID"] = "TEST-001"

    response = client.post("/predict", json=customer)

    assert response.status_code == 400
    assert "unexpected_features" in response.get_json()


def test_invalid_numeric_feature_is_rejected(client):
    customer = valid_customer()
    customer["tenure"] = "twelve"

    response = client.post("/predict", json=customer)

    assert response.status_code == 400
    assert "tenure" in response.get_json()["error"]


def test_non_json_request_is_rejected(client):
    response = client.post(
        "/predict",
        data="not json",
        content_type="text/plain",
    )

    assert response.status_code == 400
    assert "error" in response.get_json()
