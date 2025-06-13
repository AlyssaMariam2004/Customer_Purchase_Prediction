"""
Unit tests for the `main.py` module, which sets up and runs APScheduler jobs
for syncing data and retraining models periodically.
"""

import pytest
import logging
from unittest.mock import patch, MagicMock

import scheduler.main as main


def test_scheduler_initialization_positive(caplog):
    """
    Test that the scheduler starts successfully and logs the expected message.

    Mocks:
        - `sync_data`
        - `maybe_retrain_model`
        - `BlockingScheduler.start`
    """
    with patch("scheduler.main.sync_data") as mock_sync, \
         patch("scheduler.main.maybe_retrain_model") as mock_retrain, \
         patch.object(main.scheduler, "start") as mock_start, \
         caplog.at_level(logging.INFO):

        main.start_scheduler()

        assert mock_sync.called, "sync_data should be called during startup"
        assert mock_retrain.called, "maybe_retrain_model should be called during startup"
        assert mock_start.called, "scheduler.start should be called"
        assert any("Scheduler starting" in message for message in caplog.messages)


def test_scheduler_initialization_failure(caplog):
    """
    Test that an error during scheduler initialization is logged properly.

    Mocks:
        - `sync_data` to raise an Exception
    """
    with patch("scheduler.main.sync_data", side_effect=Exception("DB connection failed")), \
         patch("scheduler.main.maybe_retrain_model"), \
         patch.object(main.scheduler, "start") as mock_start, \
         caplog.at_level(logging.ERROR):

        main.start_scheduler()

        mock_start.assert_not_called()
        assert any("Error in scheduler initialization" in msg for msg in caplog.messages)
        assert any("DB connection failed" in msg for msg in caplog.messages)
