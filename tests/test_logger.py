import logging
import pytest
from unittest import mock

from app.core import logger


def test_setup_logging_positive():
    """
    Positive Test:
    Ensures logging.basicConfig is called with expected arguments.
    """
    with mock.patch("app.core.logger.logging.basicConfig") as mock_config, \
         mock.patch("app.core.logger.LOGGING_FILE_PATH", "test.log"), \
         mock.patch("app.core.logger.LOGGING_LEVEL", "DEBUG"), \
         mock.patch("app.core.logger.LOGGING_FORMAT", "%(levelname)s - %(message)s"):

        logger.setup_logging()

        mock_config.assert_called_once_with(
            filename="test.log",
            level=logging.DEBUG,
            format="%(levelname)s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )


def test_setup_logging_invalid_level_defaults_to_info():
    """
    Negative Test:
    If LOGGING_LEVEL is invalid, should default to logging.INFO.
    """
    with mock.patch("app.core.logger.logging.basicConfig") as mock_config, \
         mock.patch("app.core.logger.LOGGING_FILE_PATH", "test.log"), \
         mock.patch("app.core.logger.LOGGING_LEVEL", "INVALID"), \
         mock.patch("app.core.logger.LOGGING_FORMAT", "%(message)s"):

        logger.setup_logging()

        mock_config.assert_called_once_with(
            filename="test.log",
            level=logging.INFO,  # default fallback
            format="%(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
