"""
config.py

Centralized configuration management for the customer purchase prediction system.
Reads values from a `config.ini` file using RawConfigParser and provides constants
for file paths, database connections, logging, and retraining parameters.
"""

import os
from configparser import RawConfigParser

# Initialize and read config.ini
config = RawConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))

def get_path(section: str, key: str) -> str:
    """
    Get absolute path from a given section and key in the config.ini file.

    Args:
        section (str): Section name in the config file.
        key (str): Key under the specified section.

    Returns:
        str: Absolute path to the file or directory.
    """
    return os.path.abspath(config.get(section, key))

# Path Configuration

CSV_PATH = get_path("paths", "csv_filename")  # Path to the primary CSV data file
MODEL_DIR = get_path("paths", "model_dir")    # Directory containing model-related files

MODEL_PATH = os.path.join(
    MODEL_DIR,
    config.get("paths", "model_filename")
)  # Full path to the serialized model file

DF_PATH = os.path.join(
    MODEL_DIR,
    config.get("paths", "df_filename")
)  # Full path to the DataFrame pickle

# Database Configuration

DB_CONFIG = {
    'host': config.get("database", "host"),
    'user': config.get("database", "user"),
    'password': config.get("database", "password"),
    'database': config.get("database", "database")
}


# Logging Configuration

LOG_FILE = config.get("logging", "log_file")            # Log file path
LOG_LEVEL = config.get("logging", "log_level")          # Logging level (e.g., INFO, DEBUG)
LOG_FORMAT = config.get("logging", "log_format")        # Log message format

# Retraining Configuration

RETRAIN_INTERVAL = int(config.get("retraining", "interval_seconds"))         # How often to retrain the model (in seconds)
ROW_GROWTH_THRESHOLD = int(config.get("retraining", "row_growth_threshold")) # Row count increase needed to trigger retraining

