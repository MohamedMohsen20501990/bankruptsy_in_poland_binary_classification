
import numpy as np
import pandas as pd
import pytest

from unittest.mock import MagicMock

import src.inference as inference_module
from src.inference import (
    InferencePipeline,
    FEATURES_X,
    MODEL_FEATURES,
)
from src.schemas import BankruptsyInput


@pytest.fixture
def mock_model(monkeypatch):
    """Mock the trained model and its predict method."""
    model = MagicMock()
    model.predict.return_value = np.array([1])

    monkeypatch.setattr(
        inference_module,
        "load_obj",
        lambda model_path: model,
    )

    return model


@pytest.fixture
def valid_input():
    """Create valid input with 64 features and a Year."""
    features = {
        f"X{i}": float(i)
        for i in range(1, 65)
    }
    features["Year"] = 2023.0

    return BankruptsyInput(features=features)


@pytest.fixture
def inference_pipeline(mock_model, tmp_path):
    """Create an inference pipeline with a mocked model."""
    model_path = tmp_path / "model.pkl"

    return InferencePipeline(model_path=model_path)


# --------------------------------------------------
# Tests for initialization
# --------------------------------------------------

def test_inference_pipeline_loads_model(
    mock_model, tmp_path, monkeypatch
):
    """Test that initialization loads the model artifact."""
    model_path = tmp_path / "model.pkl"
    loaded_paths = []

    def fake_load_obj(path):
        loaded_paths.append(path)
        return mock_model

    monkeypatch.setattr(
        inference_module,
        "load_obj",
        fake_load_obj,
    )

    pipeline = InferencePipeline(model_path=model_path)

    assert pipeline.model is mock_model
    assert loaded_paths == [model_path]


# --------------------------------------------------
# Tests for input preparation
# --------------------------------------------------

def test_prepare_input_returns_dataframe(valid_input):
    """Test that input features become a one-row DataFrame."""
    df = InferencePipeline._prepare_input(valid_input)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 65)


def test_prepare_input_has_expected_columns(valid_input):
    """Test that feature names and ordering are correct."""
    df = InferencePipeline._prepare_input(valid_input)

    assert list(df.columns) == MODEL_FEATURES
    assert df.columns[:64].tolist() == FEATURES_X
    assert df.columns[-1] == "Year"


def test_prepare_input_preserves_feature_values(valid_input):
    """Test that input values are preserved in the DataFrame."""
    df = InferencePipeline._prepare_input(valid_input)

    expected = np.array(
        [valid_input.features[name] for name in MODEL_FEATURES],
        dtype=float,
    )

    np.testing.assert_array_equal(
        df.iloc[0].to_numpy(dtype=float),
        expected,
    )


def test_prepare_input_uses_numeric_values(valid_input):
    """Test that prepared feature columns are numeric."""
    df = InferencePipeline._prepare_input(valid_input)

    assert all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in df.dtypes
    )


# --------------------------------------------------
# Tests for prediction
# --------------------------------------------------

def test_predict_returns_integer(
    inference_pipeline, valid_input, mock_model
):
    """Test that predict returns an integer class."""
    result = inference_pipeline.predict(valid_input)

    assert result == 1
    assert isinstance(result, int)


def test_predict_calls_model_with_dataframe(
    inference_pipeline, valid_input, mock_model
):
    """Test that the model receives correctly prepared input."""
    inference_pipeline.predict(valid_input)

    mock_model.predict.assert_called_once()

    model_input = mock_model.predict.call_args.args[0]

    assert isinstance(model_input, pd.DataFrame)
    assert model_input.shape == (1, 65)
    assert list(model_input.columns) == MODEL_FEATURES


def test_predict_returns_zero_for_non_bankrupt(
    inference_pipeline, valid_input, mock_model
):
    """Test that class 0 is returned correctly."""
    mock_model.predict.return_value = np.array([0])

    result = inference_pipeline.predict(valid_input)

    assert result == 0
    assert isinstance(result, int)


def test_predict_propagates_model_errors(
    inference_pipeline, valid_input, mock_model
):
    """Test that prediction errors are propagated."""
    mock_model.predict.side_effect = ValueError(
        "Prediction failed"
    )

    with pytest.raises(ValueError, match="Prediction failed"):
        inference_pipeline.predict(valid_input)


# --------------------------------------------------
# Tests for input validation
# --------------------------------------------------

def test_invalid_feature_type_is_rejected():
    """Test that a list is rejected when a dictionary is required."""
    with pytest.raises((ValueError, TypeError)):
        BankruptsyInput(features=[1.0, 2.0, 3.0])


def test_invalid_feature_count_is_rejected():
    """
    Test that an incomplete feature dictionary is rejected.

    This test assumes BankruptsyInput validates that all expected
    feature names are present. If the schema only validates the
    dictionary type, add feature-key validation to the schema.
    """
    incomplete_features = {
        "X1": 1.0,
        "X2": 2.0,
    }

    with pytest.raises(ValueError):
        BankruptsyInput(features=incomplete_features)
