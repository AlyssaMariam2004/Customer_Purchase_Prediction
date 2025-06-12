import logging
from config import LOG_FILE, LOG_LEVEL, LOG_FORMAT

def setup_logging():
    logging.basicConfig(
        filename=LOG_FILE,
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
