"""
test_trainer.py

Unit tests for the trainer module in the customer purchase prediction system.

Covers:
- Model saving logic
- Pickle cleanup
- Retraining triggers
- CSV and data handling
"""

import os
import time
import shutil
import tempfile
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from scheduler import trainer


@pytest.fixture
def sample_raw_df():
    """Fixture: Returns a sample raw DataFrame."""
    return pd.DataFrame({
        "Customer ID": ["C1", "C2"],
        "Product ID": ["P1", "P2"],
        "Quantity": [3, 5],
        "Customer Age": [25, 40],
        "Customer Gender": ["F", "M"],
        "Warehouse ID": ["W1", "W2"]
    })


@pytest.fixture
def sample_processed_df():
    """Fixture: Returns a sample processed DataFrame with cluster column."""
    return pd.DataFrame({
        "P1": [1, 0],
        "P2": [0, 1],
        "Customer Age": [25, 40],
        "Customer Gender_F": [1, 0],
        "Customer Gender_M": [0, 1],
        "Warehouse ID_W1": [1, 0],
        "Warehouse ID_W2": [0, 1],
        "Cluster": [0, 1]
    })


@pytest.fixture
def temp_model_dir():
    """Fixture: Creates a temporary model directory for cleanup tests."""
    tmp_dir = tempfile.mkdtemp()
    trainer.MODEL_DIR = tmp_dir  # Patch global config path
    yield tmp_dir
    shutil.rmtree(tmp_dir)


@patch("scheduler.trainer.joblib.dump")
def test_save_model_success(mock_dump, sample_raw_df, sample_processed_df, temp_model_dir):
    """Test that model files are saved and timestamped backup files are generated."""
    trainer.DF_PATH = os.path.join(temp_model_dir, "df.pkl")
    trainer.MODEL_PATH = os.path.join(temp_model_dir, "model.pkl")

    trainer.save_model(sample_raw_df, sample_processed_df)

    assert mock_dump.call_count == 4  # 2 for normal save, 2 for timestamped versions


def test_cleanup_old_pickles_removes_excess_files(temp_model_dir):
    """Test cleanup_old_pickles deletes older .pkl files beyond the `keep` threshold."""
    # Create 6 dummy pickle files (3 pairs)
    for i in range(3):
        open(os.path.join(temp_model_dir, f"df_202406{i}_010101.pkl"), "w").close()
        open(os.path.join(temp_model_dir, f"final_df_202406{i}_010101.pkl"), "w").close()

    trainer.cleanup_old_pickles(keep=1)

    remaining = os.listdir(temp_model_dir)
    assert len(remaining) == 2  # Only latest pair should remain


@patch("scheduler.trainer.prepare_features")
@patch("scheduler.trainer.save_model")
def test_retrain_model_success(mock_save, mock_prepare, sample_raw_df, sample_processed_df, tmp_path):
    """Test retrain_model loads CSV, prepares features, and saves model."""
    # Patch CSV_PATH with a temp CSV
    trainer.CSV_PATH = tmp_path / "data.csv"
    sample_raw_df.to_csv(trainer.CSV_PATH, index=False)

    mock_prepare.return_value = sample_processed_df

    trainer.retrain_model()

    assert mock_save.called
    assert mock_prepare.called


@patch("scheduler.trainer.retrain_model")
@patch("scheduler.trainer.time")
def test_maybe_retrain_model_triggers_on_conditions(mock_time, mock_retrain, tmp_path, sample_raw_df):
    """Test maybe_retrain_model triggers retrain if time or growth thresholds are met."""
    trainer.CSV_PATH = tmp_path / "data.csv"
    sample_raw_df.to_csv(trainer.CSV_PATH, index=False)

    # Simulate that threshold conditions are met
    trainer.last_retrain_time = time.time() - (trainer.RETRAIN_INTERVAL + 1)
    trainer.last_row_count = 0

    mock_time.time.return_value = time.time()

    trainer.maybe_retrain_model()

    assert mock_retrain.called


@patch("scheduler.trainer.retrain_model")
@patch("scheduler.trainer.time")
def test_maybe_retrain_model_skips_if_conditions_not_met(mock_time, mock_retrain, tmp_path, sample_raw_df):
    """Test maybe_retrain_model skips retraining if conditions not met."""
    trainer.CSV_PATH = tmp_path / "data.csv"
    sample_raw_df.to_csv(trainer.CSV_PATH, index=False)

    # Simulate that thresholds are not met
    trainer.last_retrain_time = time.time()
    trainer.last_row_count = len(sample_raw_df)

    mock_time.time.return_value = time.time()

    trainer.maybe_retrain_model()

    assert not mock_retrain.called
