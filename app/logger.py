import os
import logging
from app.config import LOG_FILE, LOG_LEVEL, LOG_FORMAT

def setup_logging():
    # Ensure log directory exists
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    logging.basicConfig(filename=LOG_FILE, level=LOG_LEVEL, format=LOG_FORMAT)
