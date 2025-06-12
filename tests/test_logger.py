"""
test_logger.py

Unit tests for the `app.logger` module, which sets up application-wide logging
based on parameters defined in `app.config`. Tests verify that the logger is
configured correctly and logs messages as expected.
"""

import logging
import os
import pytest

from app.logger import setup_logging
from app.config import LOGGING_FILE_PATH, LOGGING_LEVEL, LOGGING_FORMAT


@pytest.fixture(autouse=True)
def reset_logging():
    """
    Reset logging before each test to avoid handler duplication.
    """
    # Clear existing handlers before test
    logging.shutdown()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


def test_logger_configuration(tmp_path):
    """
    Test that `setup_logging()` configures the logger with the correct settings
    and writes logs to the specified file.
    """
    # Temporarily override log file path
    test_log_path = tmp_path / "test_log.log"
    
    # Patch config values dynamically
    original_file_path = os.environ.get("LOGGING_FILE_PATH")
    os.environ["LOGGING_FILE_PATH"] = str(test_log_path)

    # Set up logging using overridden path
    setup_logging()

    # Write a test log entry
    logging.info("Test log message")

    # Verify the log file was created and contains the message
    assert test_log_path.exists()
    with open(test_log_path, 'r') as log_file:
        contents = log_file.read()
        assert "Test log message" in contents

    # Clean up
    if original_file_path:
        os.environ["LOGGING_FILE_PATH"] = original_file_path


def test_log_message_format_and_level(caplog):
    """
    Verify that logs are emitted with the correct level and format.
    """
    setup_logging()

    with caplog.at_level(logging.getLevelName(LOGGING_LEVEL)):
        logging.warning("This is a warning log.")

    # Check that the log was recorded with correct level and message
    assert any("This is a warning log." in message for message in caplog.messages)
    assert any(record.levelname == "WARNING" for record in caplog.records)
