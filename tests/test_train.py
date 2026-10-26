
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

import src.train as train_module
from src.train import ModelTrainer


@pytest.fixture
def trainer(tmp_path):
    """Create a ModelTrainer with a temporary model artifact path."""
    model_trainer = ModelTrainer()
    model_trainer.model_trainer_config.trained_model_path = str(
        tmp_path / "model.pkl"
    )
    return model_trainer


@pytest.fixture
def sample_train_data():
    """Create a sample training dataset."""
    return pd.DataFrame({
        "X1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "X2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "Year": [2020] * 10,
        "Class": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    })


@pytest.fixture
def sample_test_data():
    """Create a sample test dataset."""
    return pd.DataFrame({
        "X1": [11, 12, 13, 14, 15],
        "X2": [110, 120, 130, 140, 150],
        "Year": [2021] * 5,
        "Class": [0, 0, 1, 1, 1],
    })


@pytest.fixture
def mock_grid_search(monkeypatch):
    """Mock GridSearchCV with deterministic scores."""
    instances = []
    scores = [0.60, 0.70, 0.95, 0.80, 0.65, 0.75]

    class FakeGridSearchCV:
        def __init__(
            self,
            estimator,
            param_grid,
            scoring,
            cv,
            n_jobs,
            verbose,
        ):
            self.estimator = estimator
            self.param_grid = param_grid
            self.scoring = scoring
            self.cv = cv
            self.n_jobs = n_jobs
            self.verbose = verbose

            index = len(instances)
            self.best_score_ = scores[index]
            self.best_params_ = {"model__test_param": 1}
            self.best_estimator_ = None

            instances.append(self)

        def fit(self, X, y):
            self.X_fit_ = X
            self.y_fit_ = y

            self.best_estimator_ = MagicMock()
            self.best_estimator_.predict.return_value = np.array(
                [0, 0, 1, 1, 1]
            )

            return self

    monkeypatch.setattr(
        train_module,
        "GridSearchCV",
        FakeGridSearchCV,
    )

    return instances


@pytest.fixture
def mock_save_model(monkeypatch):
    """Prevent tests from writing real model artifacts."""
    saved = {}

    def fake_save_obj(obj, file_path):
        saved["obj"] = obj
        saved["path"] = file_path

    monkeypatch.setattr(train_module, "save_obj", fake_save_obj)

    return saved


# --------------------------------------------------
# Tests for train_model()
# --------------------------------------------------

def test_train_model_returns_best_model(
    trainer,
    sample_train_data,
    sample_test_data,
    mock_grid_search,
    mock_save_model,
):
    """Test that train_model returns the selected estimator."""
    best_model = trainer.train_model(
        train_data=sample_train_data,
        test_data=sample_test_data,
        target="Class",
    )

    assert best_model is not None
    assert hasattr(best_model, "predict")


def test_train_model_evaluates_all_six_models(
    trainer,
    sample_train_data,
    sample_test_data,
    mock_grid_search,
    mock_save_model,
):
    """Test that all six configured models are evaluated."""
    trainer.train_model(
        train_data=sample_train_data,
        test_data=sample_test_data,
        target="Class",
    )

    expected_models = {
        "Logistic Regression",
        "Decision tree",
        "Random Forest",
        "Gradient Boosting",
        "KNN",
        "XGBOOST",
    }

    assert len(mock_grid_search) == 6
    assert set(trainer.report.keys()) == expected_models


def test_train_model_uses_recall_scoring(
    trainer,
    sample_train_data,
    sample_test_data,
    mock_grid_search,
    mock_save_model,
):
    """Test that GridSearchCV uses recall as its scoring metric."""
    trainer.train_model(
        train_data=sample_train_data,
        test_data=sample_test_data,
        target="Class",
    )

    assert all(
        search.scoring == "recall"
        for search in mock_grid_search
    )


def test_train_model_stores_report(
    trainer,
    sample_train_data,
    sample_test_data,
    mock_grid_search,
    mock_save_model,
):
    """Test that CV scores and best parameters are recorded."""
    trainer.train_model(
        train_data=sample_train_data,
        test_data=sample_test_data,
        target="Class",
    )

    assert trainer.report

    for model_info in trainer.report.values():
        assert "best_recall" in model_info
        assert "best params" in model_info
        assert isinstance(model_info["best_recall"], float)


def test_train_model_selects_best_model(
    trainer,
    sample_train_data,
    sample_test_data,
    mock_grid_search,
    mock_save_model,
):
    """Test that the model with the highest CV recall is selected."""
    trainer.train_model(
        train_data=sample_train_data,
        test_data=sample_test_data,
        target="Class",
    )

    assert trainer.best_model_name == "Random Forest"
    assert trainer.report["Random Forest"]["best_recall"] == 0.95


def test_train_model_calculates_test_recall(
    trainer,
    sample_train_data,
    sample_test_data,
    mock_grid_search,
    mock_save_model,
):
    """Test that test recall is calculated and stored."""
    trainer.train_model(
        train_data=sample_train_data,
        test_data=sample_test_data,
        target="Class",
    )

    # Test labels:      [0, 0, 1, 1, 1]
    # Mock predictions: [0, 0, 1, 1, 1]
    assert trainer.best_test_recall == pytest.approx(1.0)


def test_train_model_saves_best_model(
    trainer,
    sample_train_data,
    sample_test_data,
    mock_grid_search,
    mock_save_model,
):
    """Test that the selected model is saved to the configured path."""
    best_model = trainer.train_model(
        train_data=sample_train_data,
        test_data=sample_test_data,
        target="Class",
    )

    assert mock_save_model["obj"] is best_model
    assert (
        mock_save_model["path"]
        == trainer.model_trainer_config.trained_model_path
    )


def test_train_model_missing_target(
    trainer,
    sample_train_data,
    sample_test_data,
):
    """Test that a missing training target raises KeyError."""
    train_data = sample_train_data.drop(columns=["Class"])

    with pytest.raises(KeyError):
        trainer.train_model(
            train_data=train_data,
            test_data=sample_test_data,
            target="Class",
        )


def test_train_model_missing_test_target(
    trainer,
    sample_train_data,
    sample_test_data,
):
    """Test that a missing test target raises KeyError."""
    test_data = sample_test_data.drop(columns=["Class"])

    with pytest.raises(KeyError):
        trainer.train_model(
            train_data=sample_train_data,
            test_data=test_data,
            target="Class",
        )
