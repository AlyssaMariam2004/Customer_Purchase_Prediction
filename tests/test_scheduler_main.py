import pytest
from unittest.mock import patch, MagicMock, call
from scheduler import main


@patch("scheduler.main.sync_data")
@patch("scheduler.main.maybe_retrain_model")
@patch("scheduler.main.BlockingScheduler")
def test_main_scheduler_starts_successfully(
    mock_scheduler_cls, mock_retrain, mock_sync
):
    """
    Positive Test:
    Should add two jobs and start the scheduler successfully.
    """
    mock_scheduler = MagicMock()
    mock_scheduler_cls.return_value = mock_scheduler

    main.start_scheduler()

    # Assert two jobs are scheduled
    assert mock_scheduler.add_job.call_count == 2

    # Assert both functions are run once manually
    mock_sync.assert_called_once()
    mock_retrain.assert_called_once()

    # Scheduler should be started
    mock_scheduler.start.assert_called_once()


@patch("scheduler.main.sync_data", side_effect=Exception("Boom"))
@patch("scheduler.main.maybe_retrain_model")
@patch("scheduler.main.BlockingScheduler")
@patch("scheduler.main.logging")
def test_main_scheduler_job_add_failure(
    mock_logging, mock_scheduler_cls, mock_retrain, mock_sync
):
    """
    Negative Test:
    Should catch exception from sync_data and log an error.
    """
    mock_scheduler = MagicMock()
    mock_scheduler_cls.return_value = mock_scheduler

    main.start_scheduler()

    # It should have attempted to run sync_data and raised
    mock_sync.assert_called_once()

    # It should have logged the error
    mock_logging.error.assert_any_call("Error in scheduler initialization: Boom")

