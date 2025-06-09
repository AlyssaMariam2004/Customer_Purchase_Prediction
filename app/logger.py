"""
Logging setup module.

Initializes the logging configuration for the application based on values
defined in the config file. Ensures the log directory exists before setting up.
"""

import os
import logging
from app.config import LOG_FILE, LOG_LEVEL, LOG_FORMAT

def setup_logging():
    """
    Configure the logging system.

    - Creates the log directory if it doesn't exist.
    - Sets up logging to write to the configured log file with the specified
      log level and format.
    """
    # Ensure log directory exists
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    # Set up logging with file, level, and format
    logging.basicConfig(filename=LOG_FILE, level=LOG_LEVEL, format=LOG_FORMAT)

