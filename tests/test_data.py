
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal
from sklearn.model_selection import train_test_split

from src.data import DataHandler


@pytest.fixture
def data_handler(tmp_path):
    """Create a DataHandler with temporary output paths."""
    handler = DataHandler()

    handler.data_config.train_data_path = str(
        tmp_path / "artifacts" / "train_data.csv"
    )
    handler.data_config.test_data_path = str(
        tmp_path / "artifacts" / "test_data.csv"
    )

    return handler


@pytest.fixture
def sample_dataframe():
    """Create a sample dataset for testing."""
    return pd.DataFrame({
        "id": range(1, 101),
        "Class": [0] * 80 + [1] * 20,
        "Year": [2020] * 100,
        "feature_1": range(100),
        "feature_2": [float(i) for i in range(100)],
    })


@pytest.fixture
def sample_csv(tmp_path):
    """Create a temporary CSV file for testing."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "Class": ["0", "1", "0", "1"],
        "Year": ["2020", "2021", "2020", "2021"],
        "feature_1": [10.5, 20.5, 30.5, 40.5],
    })

    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


# --------------------------------------------------
# Tests for read_csv_data()
# --------------------------------------------------

def test_read_csv_data_success(data_handler, sample_csv):
    """Test successful CSV loading and basic transformations."""
    df = data_handler.read_csv_data(str(sample_csv))

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4

    # The id column should be removed.
    assert "id" not in df.columns

    # Target and year should be integers.
    assert pd.api.types.is_integer_dtype(df["Class"])
    assert pd.api.types.is_integer_dtype(df["Year"])

    # Index should be reset.
    assert list(df.index) == [0, 1, 2, 3]


def test_read_csv_data_missing_file(data_handler, tmp_path):
    """Test that a missing CSV file raises an exception."""
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        data_handler.read_csv_data(str(missing_path))


def test_read_csv_data_invalid_integer_column(data_handler, tmp_path):
    """Test that invalid Class or Year values raise an exception."""
    df = pd.DataFrame({
        "id": [1, 2],
        "Class": ["valid", "invalid"],
        "Year": ["2020", "2021"],
    })

    csv_path = tmp_path / "invalid.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises((ValueError, TypeError)):
        data_handler.read_csv_data(str(csv_path))


def test_read_csv_data_missing_required_column(data_handler, tmp_path):
    """Test that missing required columns raise an exception."""
    df = pd.DataFrame({
        "id": [1, 2],
        "Class": [0, 1],
    })

    csv_path = tmp_path / "missing_column.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises((KeyError, ValueError, TypeError)):
        data_handler.read_csv_data(str(csv_path))


# --------------------------------------------------
# Tests for split_and_save()
# --------------------------------------------------

def test_split_and_save_returns_dataframes(
    data_handler, sample_dataframe
):
    """Test that splitting returns training and test DataFrames."""
    train_data, test_data = data_handler.split_and_save(
        sample_dataframe
    )

    assert isinstance(train_data, pd.DataFrame)
    assert isinstance(test_data, pd.DataFrame)

    # 80% training, 20% testing.
    assert len(train_data) == 80
    assert len(test_data) == 20


def test_split_and_save_creates_files(
    data_handler, sample_dataframe
):
    """Test that training and test CSV files are created."""
    data_handler.split_and_save(sample_dataframe)

    train_path = data_handler.data_config.train_data_path
    test_path = data_handler.data_config.test_data_path

    import os

    assert os.path.isfile(train_path)
    assert os.path.isfile(test_path)

    # Confirm the saved files can be loaded.
    saved_train = pd.read_csv(train_path)
    saved_test = pd.read_csv(test_path)

    assert len(saved_train) == 80
    assert len(saved_test) == 20


def test_split_and_save_preserves_all_rows(
    data_handler, sample_dataframe
):
    """Test that splitting neither loses nor duplicates rows."""
    train_data, test_data = data_handler.split_and_save(
        sample_dataframe
    )

    combined = pd.concat(
        [train_data, test_data],
        ignore_index=True,
    )

    expected = sample_dataframe.reset_index(drop=True)

    assert_frame_equal(
        combined.sort_values("feature_1").reset_index(drop=True),
        expected,
    )


def test_split_and_save_stratifies_target(
    data_handler, sample_dataframe
):
    """Test that target proportions are preserved approximately."""
    train_data, test_data = data_handler.split_and_save(
        sample_dataframe
    )

    original_ratio = sample_dataframe["Class"].mean()
    train_ratio = train_data["Class"].mean()
    test_ratio = test_data["Class"].mean()

    assert train_ratio == pytest.approx(original_ratio, abs=0.02)
    assert test_ratio == pytest.approx(original_ratio, abs=0.02)


def test_split_and_save_is_reproducible(
    data_handler, sample_dataframe
):
    """Test that random_state=42 gives reproducible splits."""
    train_1, test_1 = data_handler.split_and_save(
        sample_dataframe
    )

    train_2, test_2 = train_test_split(
        sample_dataframe,
        test_size=0.2,
        random_state=42,
        stratify=sample_dataframe["Class"],
    )

    assert_frame_equal(train_1, train_2)
    assert_frame_equal(test_1, test_2)


def test_split_and_save_invalid_target(
    data_handler, sample_dataframe
):
    """Test that a missing target column raises an exception."""
    df = sample_dataframe.drop(columns=["Class"])

    with pytest.raises(KeyError):
        data_handler.split_and_save(df)
