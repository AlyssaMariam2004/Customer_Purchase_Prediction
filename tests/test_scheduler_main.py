"""
test_scheduler_main.py

Unit tests for the scheduler.main module responsible for scheduling periodic
data ingestion and model retraining using APScheduler.
"""

import pytest
from unittest.mock import patch, MagicMock, call
import scheduler.main as scheduler_main


@patch("scheduler.main.sync_data")
@patch("scheduler.main.maybe_retrain_model")
@patch("scheduler.main.BlockingScheduler")
def test_scheduler_jobs_registered(mock_scheduler_class, mock_retrain, mock_sync):
    """
    Test that scheduler registers sync_data and maybe_retrain_model jobs correctly.
    """
    mock_scheduler_instance = MagicMock()
    mock_scheduler_class.return_value = mock_scheduler_instance

    # Trigger job registration
    with patch("scheduler.main.__name__", "__main__"):  # Simulate script as main
        scheduler_main.sync_data = mock_sync
        scheduler_main.maybe_retrain_model = mock_retrain
        scheduler_main.scheduler = mock_scheduler_instance
        
        scheduler_main.sync_data()
        scheduler_main.maybe_retrain_model()

        # Simulate scheduler start
        scheduler_main.scheduler.start()

    # Verify both jobs were added
    assert mock_scheduler_instance.add_job.call_count == 2
    add_job_calls = mock_scheduler_instance.add_job.call_args_list

    # Validate call args contain correct function references
    assert add_job_calls[0].args[0] == mock_sync
    assert add_job_calls[1].args[0] == mock_retrain

    # Check that start was called
    mock_scheduler_instance.start.assert_called_once()


@patch("scheduler.main.sync_data", side_effect=Exception("Sync failed"))
@patch("scheduler.main.maybe_retrain_model", side_effect=Exception("Train failed"))
@patch("scheduler.main.BlockingScheduler")
def test_scheduler_job_exceptions_logged(mock_scheduler_class, mock_retrain, mock_sync, caplog):
    """
    Test logging of exceptions raised during scheduler setup.
    """
    mock_scheduler = MagicMock()
    mock_scheduler_class.return_value = mock_scheduler

    with patch("scheduler.main.__name__", "__main__"):
        scheduler_main.sync_data = mock_sync
        scheduler_main.maybe_retrain_model = mock_retrain
        scheduler_main.scheduler = mock_scheduler

        try:
            scheduler_main.sync_data()
        except Exception:
            pass

        try:
            scheduler_main.maybe_retrain_model()
        except Exception:
            pass

    assert "Sync failed" in caplog.text or "Train failed" in caplog.text
