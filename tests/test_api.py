import pytest

from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_prediction(client):

    payload = {
        "age": 32,
        "monthly_income": 50000,
        "employment_type": "salaried",
        "credit_score": 720,
        "existing_emi": 10000,
        "loan_amount": 300000,
        "loan_tenure": 24,
        "previous_default": 0,
        "loan_to_income": 0.5,
        "emi_to_income": 0.2
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert 0 <= data["default_probability"] <= 1

    assert data["prediction"] in [0, 1]

    assert data["risk_class"] in [
        "LOW",
        "MEDIUM",
        "HIGH"
    ]


def test_invalid_input(client):

    payload = {
        "age": "wrong",
        "monthly_income": 50000,
        "employment_type": "salaried",
        "credit_score": 720,
        "existing_emi": 10000,
        "loan_amount": 300000,
        "loan_tenure": 24,
        "previous_default": 0,
        "loan_to_income": 0.5,
        "emi_to_income": 0.2
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422