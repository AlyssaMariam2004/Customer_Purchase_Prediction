"""
logger.py

This module sets up the logging configuration for the application.
It reads log file location, log level, and format from the config settings.
"""

import logging
from app.config import LOGGING_FILE_PATH, LOGGING_LEVEL, LOGGING_FORMAT

def setup_logging() -> None:
    """
    Configures the logging for the application.

    Reads configuration from app.config and sets up:
    - Log file destination
    - Logging level (e.g., DEBUG, INFO, WARNING)
    - Logging format
    - Date/time formatting
    """
    logging.basicConfig(
        filename=LOGGING_FILE_PATH,
        level=getattr(logging, LOGGING_LEVEL.upper(), logging.INFO),
        format=LOGGING_FORMAT,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

