"""
logger.py

This module sets up the global logging configuration for the application
using parameters defined in the config module.
"""

import logging
from config import LOG_FILE, LOG_LEVEL, LOG_FORMAT

def setup_logging():
    """
    Configures the global logging settings for the application.

    Uses parameters from the config file to:
    - Set the log file destination.
    - Set the log level (e.g., DEBUG, INFO, WARNING).
    - Format the log messages.
    """
    logging.basicConfig(
        filename=LOG_FILE,                         # Path to the log file
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),  # Log level with fallback to INFO
        format=LOG_FORMAT,                         # Log message format
        datefmt='%Y-%m-%d %H:%M:%S'                # Timestamp format
    )

