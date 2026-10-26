
import pandas as pd
import numpy as np
import pytest

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from pandas.testing import assert_frame_equal

from src.preprocess import Preprocessor


@pytest.fixture
def preprocessor(tmp_path):
    """Create a Preprocessor with temporary artifact paths."""
    processor = Preprocessor()

    processor.preprocessor_config.preprocessor_path = str(
        tmp_path / "preprocessor.pkl"
    )
    processor.preprocessor_config.transformed_train_data_path = str(
        tmp_path / "transformed_train_data.parquet"
    )
    processor.preprocessor_config.transformed_test_data_path = str(
        tmp_path / "transformed_test_data.parquet"
    )

    return processor


@pytest.fixture
def sample_train_data():
    """Create sample training data with numeric features and a target."""
    return pd.DataFrame({
        "X1": [10.0, 20.0, np.nan, 40.0, 50.0, 60.0],
        "X2": [100.0, 200.0, 300.0, np.nan, 500.0, 600.0],
        "X3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "Year": [2018, 2019, 2020, 2021, 2022, 2023],
        "Class": [0, 0, 0, 1, 1, 1],
    })


@pytest.fixture
def sample_test_data():
    """Create sample test data with missing numeric values."""
    return pd.DataFrame({
        "X1": [15.0, np.nan, 45.0],
        "X2": [150.0, 350.0, np.nan],
        "X3": [1.5, 3.5, 4.5],
        "Year": [2019, 2021, 2022],
        "Class": [0, 1, 1],
    })


@pytest.fixture
def csv_paths(tmp_path, sample_train_data, sample_test_data):
    """Save sample training and test data to temporary CSV files."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    sample_train_data.to_csv(train_path, index=False)
    sample_test_data.to_csv(test_path, index=False)

    return train_path, test_path


# --------------------------------------------------
# Tests for build_preprocessor()
# --------------------------------------------------

def test_build_preprocessor_returns_column_transformer(preprocessor):
    """Test that the preprocessor is a ColumnTransformer."""
    result = preprocessor.build_preprocessor(["X1", "X2", "X3"])

    assert isinstance(result, ColumnTransformer)



def test_build_preprocessor_pipeline_structure(preprocessor):
    """Test that numerical features use imputation and scaling."""
    result = preprocessor.build_preprocessor(["X1", "X2", "X3"])

    # Find the numerical transformer by its name.
    numerical_pipeline = next(
        transformer
        for name, transformer, columns in result.transformers
        if name == "num"
    )

    assert isinstance(numerical_pipeline, Pipeline)

    assert isinstance(
        numerical_pipeline.named_steps["imputer"],
        SimpleImputer,
    )
    assert numerical_pipeline.named_steps["imputer"].strategy == "median"

    assert isinstance(
        numerical_pipeline.named_steps["scaler"],
        RobustScaler,
    )


def test_build_preprocessor_remainder_passthrough(preprocessor):
    """Test that non-numerical-feature columns are passed through."""
    result = preprocessor.build_preprocessor(["X1", "X2"])

    assert result.remainder == "passthrough"


# --------------------------------------------------
# Tests for start_data_preprocessing()
# --------------------------------------------------

def test_start_data_preprocessing_returns_dataframes(
    preprocessor, csv_paths
):
    """Test that preprocessing returns training and test DataFrames."""
    train_path, test_path = csv_paths

    train_df, test_df, artifact_path = (
        preprocessor.start_data_preprocessing(
            train_path=train_path,
            test_path=test_path,
        )
    )

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert artifact_path == preprocessor.preprocessor_config.preprocessor_path


def test_start_data_preprocessing_preserves_rows(
    preprocessor, csv_paths, sample_train_data, sample_test_data
):
    """Test that preprocessing preserves the number of rows."""
    train_path, test_path = csv_paths

    train_df, test_df, _ = preprocessor.start_data_preprocessing(
        train_path=train_path,
        test_path=test_path,
    )

    assert len(train_df) == len(sample_train_data)
    assert len(test_df) == len(sample_test_data)


def test_start_data_preprocessing_preserves_target(
    preprocessor, csv_paths, sample_train_data, sample_test_data
):
    """Test that target labels remain unchanged."""
    train_path, test_path = csv_paths

    train_df, test_df, _ = preprocessor.start_data_preprocessing(
        train_path=train_path,
        test_path=test_path,
    )

    assert train_df["Class"].tolist() == sample_train_data["Class"].tolist()
    assert test_df["Class"].tolist() == sample_test_data["Class"].tolist()


def test_start_data_preprocessing_handles_missing_values(
    preprocessor, csv_paths
):
    """Test that missing numerical values are imputed."""
    train_path, test_path = csv_paths

    train_df, test_df, _ = preprocessor.start_data_preprocessing(
        train_path=train_path,
        test_path=test_path,
    )

    # All numerical features should be complete after imputation.
    for column in ["X1", "X2", "X3"]:
        assert train_df[column].notna().all()
        assert test_df[column].notna().all()


def test_start_data_preprocessing_scales_features(
    preprocessor, csv_paths
):
    """Test that numerical features are scaled."""
    train_path, test_path = csv_paths

    train_df, test_df, _ = preprocessor.start_data_preprocessing(
        train_path=train_path,
        test_path=test_path,
    )

    # RobustScaler centers the training median at approximately zero.
    for column in ["X1", "X2", "X3"]:
        assert train_df[column].median() == pytest.approx(0.0)

    # The test set is transformed using the training-fitted scaler.
    assert np.isfinite(test_df[["X1", "X2", "X3"]].to_numpy()).all()


def test_start_data_preprocessing_preserves_passthrough_feature(
    preprocessor, csv_paths, sample_train_data, sample_test_data
):
    """Test that Year remains unchanged by preprocessing."""
    train_path, test_path = csv_paths

    train_df, test_df, _ = preprocessor.start_data_preprocessing(
        train_path=train_path,
        test_path=test_path,
    )

    assert train_df["Year"].tolist() == sample_train_data["Year"].tolist()
    assert test_df["Year"].tolist() == sample_test_data["Year"].tolist()


def test_start_data_preprocessing_saves_artifacts(
    preprocessor, csv_paths
):
    """Test that the preprocessor and transformed datasets are saved."""
    train_path, test_path = csv_paths

    preprocessor.start_data_preprocessing(
        train_path=train_path,
        test_path=test_path,
    )

    config = preprocessor.preprocessor_config

    assert Path(config.preprocessor_path).is_file()
    assert Path(config.transformed_train_data_path).is_file()
    assert Path(config.transformed_test_data_path).is_file()

    saved_train = pd.read_parquet(config.transformed_train_data_path)
    saved_test = pd.read_parquet(config.transformed_test_data_path)

    assert len(saved_train) == 6
    assert len(saved_test) == 3


def test_start_data_preprocessing_missing_train_file(
    preprocessor, tmp_path
):
    """Test that a missing training file raises an exception."""
    missing_train = tmp_path / "missing_train.csv"
    test_path = tmp_path / "test.csv"

    pd.DataFrame({
        "X1": [1.0, 2.0],
        "Class": [0, 1],
    }).to_csv(test_path, index=False)

    with pytest.raises(FileNotFoundError):
        preprocessor.start_data_preprocessing(
            train_path=missing_train,
            test_path=test_path,
        )


def test_start_data_preprocessing_missing_target(
    preprocessor, tmp_path
):
    """Test that missing target columns raise an exception."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    invalid_df = pd.DataFrame({
        "X1": [1.0, 2.0, 3.0],
        "Year": [2020, 2021, 2022],
    })

    invalid_df.to_csv(train_path, index=False)
    invalid_df.to_csv(test_path, index=False)

    with pytest.raises(KeyError):
        preprocessor.start_data_preprocessing(
            train_path=train_path,
            test_path=test_path,
        )
