import os
import time
import pytest
import pandas as pd
import joblib
from unittest import mock
from scheduler import trainer


@pytest.fixture
def raw_df():
    """Returns a sample raw DataFrame with customer data."""
    return pd.DataFrame({
        "Customer ID": ["C1", "C2"],
        "Product ID": ["P1", "P1"],
        "Quantity": [3, 2],
        "Customer Age": [25, 35],
        "Customer Gender": ["M", "F"],
        "Warehouse ID": ["W1", "W2"]
    })


@pytest.fixture
def processed_df():
    """Returns a processed DataFrame with clustering results."""
    return pd.DataFrame({
        "Customer Age": [25, 35],
        "Cluster": [0, 1]
    })


def test_save_model_success(tmp_path, raw_df, processed_df):
    """
    Positive test: save_model should write model files when given valid DataFrames.
    """
    with mock.patch("scheduler.trainer.DF_PATH", tmp_path / "df.pkl"), \
         mock.patch("scheduler.trainer.MODEL_PATH", tmp_path / "model.pkl"), \
         mock.patch("scheduler.trainer.MODEL_DIR", tmp_path):

        trainer.save_model(raw_df, processed_df)

        df_file = tmp_path / "df.pkl"
        model_file = tmp_path / "model.pkl"
        assert df_file.exists()
        assert model_file.exists()

        # Check that the content is a DataFrame
        loaded_df = joblib.load(df_file)
        assert isinstance(loaded_df, pd.DataFrame)


def test_save_model_invalid_input(raw_df):
    """
    Negative test: save_model should log errors if inputs are invalid types.
    """
    # Case 1: raw_data is invalid
    with mock.patch("scheduler.trainer.logging.error") as mock_log:
        trainer.save_model("invalid", raw_df)
        assert mock_log.called
        assert "raw_data must be a pandas DataFrame" in str(mock_log.call_args)

    # Case 2: processed_data is invalid
    with mock.patch("scheduler.trainer.logging.error") as mock_log:
        trainer.save_model(raw_df, "invalid")
        assert mock_log.called
        assert "processed_data must be a pandas DataFrame" in str(mock_log.call_args)




def test_cleanup_old_pickles(tmp_path):
    """
    Positive test: cleanup_old_pickles should delete old pickle files, keeping only recent ones.
    """
    for i in range(4):
        for suffix in ["df", "final_df"]:
            file = tmp_path / f"{suffix}_2024060{i}.pkl"
            file.write_text("dummy")

    with mock.patch("scheduler.trainer.MODEL_DIR", tmp_path):
        trainer.cleanup_old_pickles(keep=1)

        remaining = list(tmp_path.glob("*.pkl"))
        assert len(remaining) == 2  # Should keep 2 files: 1 df_ and 1 final_df_


def test_retrain_model_executes(tmp_path, raw_df):
    """
    Positive test: retrain_model should save model if CSV exists and is valid.
    """
    csv_path = tmp_path / "data.csv"
    raw_df.to_csv(csv_path, index=False)

    with mock.patch("scheduler.trainer.CSV_PATH", str(csv_path)), \
         mock.patch("scheduler.trainer.prepare_features", return_value=raw_df), \
         mock.patch("scheduler.trainer.save_model") as mock_save:
        trainer.retrain_model()
        mock_save.assert_called_once()


def test_retrain_model_missing_file(tmp_path):
    """
    Negative test: retrain_model should log warning if CSV doesn't exist.
    """
    with mock.patch("scheduler.trainer.CSV_PATH", tmp_path / "missing.csv"), \
         mock.patch("scheduler.trainer.logging") as mock_log:
        trainer.retrain_model()
        assert mock_log.warning.called


def test_maybe_retrain_model_threshold_time(tmp_path, raw_df):
    """
    Positive test: maybe_retrain_model should trigger retraining if enough time passed.
    """
    csv_path = tmp_path / "data.csv"
    raw_df.to_csv(csv_path, index=False)

    with mock.patch("scheduler.trainer.CSV_PATH", str(csv_path)), \
         mock.patch("scheduler.trainer.last_retrain_time", time.time() - 1000), \
         mock.patch("scheduler.trainer.RETRAIN_INTERVAL", 100), \
         mock.patch("scheduler.trainer.last_row_count", 0), \
         mock.patch("scheduler.trainer.retrain_model") as mock_retrain:
        trainer.maybe_retrain_model()
        mock_retrain.assert_called_once()


def test_maybe_retrain_model_threshold_growth(tmp_path, raw_df):
    """
    Positive test: maybe_retrain_model should retrain if row growth exceeds threshold.
    """
    csv_path = tmp_path / "data.csv"
    raw_df.to_csv(csv_path, index=False)

    with mock.patch("scheduler.trainer.CSV_PATH", str(csv_path)), \
         mock.patch("scheduler.trainer.last_retrain_time", time.time()), \
         mock.patch("scheduler.trainer.last_row_count", 0), \
         mock.patch("scheduler.trainer.ROW_GROWTH_THRESHOLD", 1), \
         mock.patch("scheduler.trainer.retrain_model") as mock_retrain:
        trainer.maybe_retrain_model()
        mock_retrain.assert_called_once()

