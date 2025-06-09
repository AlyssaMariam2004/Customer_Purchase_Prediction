import os
import logging
from app.config import LOG_FILE, LOG_LEVEL, LOG_FORMAT

def setup_logging():
    """
    Configure the logging system.

    - Creates the log directory if it doesn't exist.
    - Sets up logging to write logs both to the configured log file and console
      with the specified log level and format.
    """
    # Ensure log directory exists
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    # Remove all existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)

    # File handler - logs to file
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console handler - logs to terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
