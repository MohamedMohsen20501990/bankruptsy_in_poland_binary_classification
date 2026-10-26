
import pytest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import src.main as main_module
from src.main import app


@pytest.fixture
def mock_inference_pipeline(monkeypatch):
    """Mock InferencePipeline to avoid loading a real model."""
    pipeline = MagicMock()
    pipeline.predict.return_value = 1

    monkeypatch.setattr(
        main_module,
        "InferencePipeline",
        lambda model_path: pipeline,
    )

    return pipeline


@pytest.fixture
def client(mock_inference_pipeline):
    """Create a test client and run the application lifespan."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload():
    """Create a valid prediction request payload."""
    features = {
        f"X{i}": float(i)
        for i in range(1, 65)
    }
    features["Year"] = 2023.0

    return {"features": features}


# --------------------------------------------------
# Tests for application startup and shutdown
# --------------------------------------------------

def test_app_starts_and_loads_inference_pipeline(
    mock_inference_pipeline,
):
    """Test that the lifespan initializes the inference pipeline."""
    with TestClient(app):
        assert hasattr(app.state, "inference_pipeline")
        assert (
            app.state.inference_pipeline
            is mock_inference_pipeline
        )


def test_app_lifespan_logs_startup_and_shutdown(
    mock_inference_pipeline,
    monkeypatch,
):
    """Test that startup and shutdown messages are logged."""
    info_mock = MagicMock()

    monkeypatch.setattr(main_module.logger, "info", info_mock)

    with TestClient(app):
        pass

    messages = [
        call.args[0]
        for call in info_mock.call_args_list
    ]

    assert "Loading bankruptcy prediction model" in messages
    assert "Model loaded successfully" in messages
    assert "Application shutting down" in messages


# --------------------------------------------------
# Tests for POST /predict
# --------------------------------------------------

def test_predict_success(client, mock_inference_pipeline, valid_payload):
    """Test a successful bankruptcy prediction."""
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "prediction": 1,
        "message": "Prediction completed successfully!",
    }

    mock_inference_pipeline.predict.assert_called_once()


def test_predict_returns_class_zero(
    client,
    mock_inference_pipeline,
    valid_payload,
):
    """Test a successful prediction of class 0."""
    mock_inference_pipeline.predict.return_value = 0

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json()["prediction"] == 0
    assert response.json()["success"] is True


def test_predict_invalid_payload_returns_422(client):
    """Test that a request with an invalid payload is rejected."""
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0]},
    )

    assert response.status_code == 422


def test_predict_missing_features_returns_422(client):
    """Test that a request without features is rejected."""
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_predict_missing_feature_keys_returns_422(client):
    """
    Test incomplete feature dictionaries.

    This expects BankruptsyInput to validate the required feature keys.
    """
    response = client.post(
        "/predict",
        json={"features": {"X1": 1.0, "X2": 2.0}},
    )

    assert response.status_code == 422


def test_predict_model_failure_returns_500(
    client,
    mock_inference_pipeline,
    valid_payload,
):
    """Test that inference failures return a generic HTTP 500."""
    mock_inference_pipeline.predict.side_effect = RuntimeError(
        "Internal model details"
    )

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 500
    assert response.json() == {
        "detail": "An internal error occurred during prediction"
    }


def test_predict_passes_validated_input_to_pipeline(
    client,
    mock_inference_pipeline,
    valid_payload,
):
    """Test that the validated Pydantic input reaches the pipeline."""
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200

    mock_inference_pipeline.predict.assert_called_once()

    call_kwargs = mock_inference_pipeline.predict.call_args.kwargs
    input_data = call_kwargs["input_data"]

    assert input_data.features["X1"] == 1.0
    assert input_data.features["X64"] == 64.0
    assert input_data.features["Year"] == 2023.0


# --------------------------------------------------
# Tests for unsupported routes
# --------------------------------------------------

def test_root_route_not_defined(client):
    """Test that the root route returns 404 when not configured."""
    response = client.get("/")

    assert response.status_code == 404
