"""
config_loader.py

This module handles configuration loading from a `.ini` file.
It extracts paths, database credentials, logging preferences, model parameters,
and retraining thresholds to be used throughout the application.
"""

import os
from configparser import RawConfigParser

# Initialize and read the configuration file
config = RawConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))

def get_absolute_path(section: str, key: str) -> str:
    """
    Retrieve the absolute path for a given section and key from the config file.

    Args:
        section (str): The section in the config file.
        key (str): The specific key within the section.

    Returns:
        str: Absolute path corresponding to the key.
    """
    return os.path.abspath(config.get(section, key))


# File and model paths
CSV_FILE_PATH = get_absolute_path("paths", "csv_path")
MODEL_DIRECTORY = get_absolute_path("paths", "model_dir")
MODEL_FILE_PATH = get_absolute_path("paths", "model_path")
DATAFRAME_PATH = get_absolute_path("paths", "df_path")

# Database configuration
DATABASE_CONFIG = {
    'host': config.get("database", "host"),
    'user': config.get("database", "user"),
    'password': config.get("database", "password"),
    'database': config.get("database", "database")
}

# Logging configuration
LOGGING_FILE_PATH = config.get("logging", "log_file")
LOGGING_LEVEL = config.get("logging", "log_level")
LOGGING_FORMAT = config.get("logging", "log_format")

# Model retraining configuration
MODEL_RETRAIN_INTERVAL = int(config.get("retrain", "interval"))  # in seconds or minutes
ROW_GROWTH_THRESHOLD = int(config.get("retrain", "row_growth_threshold"))  # trigger retraining if rows exceed this threshold

# Default recommendation output size
DEFAULT_TOP_N = int(config.get("model", "top_n"))
