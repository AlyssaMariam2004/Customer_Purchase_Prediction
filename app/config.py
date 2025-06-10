"""
Configuration module for the project.

This script reads settings from a 'config.ini' file and exposes them as constants
to be used across the application. It handles paths, database credentials,
retraining logic parameters, and logging configurations.

Usage:
    from config import DB_CONFIG, CSV_PATH, RETRAIN_INTERVAL, ...
"""

from configparser import RawConfigParser
import os

# Initialize and read configuration from config.ini
config = RawConfigParser()
config.read(os.path.join(os.path.dirname(__file__), "config.ini"))

# Base directory of the project (one level above this file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Database Configuration
# Dictionary containing connection details for MySQL or similar database
DB_CONFIG = {
    "host": config["database"]["host"],
    "user": config["database"]["user"],
    "password": config["database"]["password"],
    "database": config["database"]["database"]
}

# Full path to the dataset CSV file, based on config and project structure
CSV_PATH = os.path.join(BASE_DIR, "data", config["paths"]["csv_filename"])

# Retraining parameters
# Interval in seconds for periodic retraining of the model
RETRAIN_INTERVAL = int(config["retraining"]["interval_seconds"])

# Threshold of new rows required to trigger retraining
ROW_GROWTH_THRESHOLD = int(config["retraining"]["row_growth_threshold"])

# Logging Configuration
# Path to the log file
LOG_FILE = os.path.join(BASE_DIR, config["logging"]["log_file"])

# Log level (e.g., DEBUG, INFO, WARNING)
LOG_LEVEL = config["logging"]["log_level"]

# Format string for log messages
LOG_FORMAT = config["logging"]["log_format"]

#Default recommendations to be shown
TOP_N_DEFAULT = int(config["model"]["top_n"])

MODEL_DIR = os.path.join(BASE_DIR, config["paths"]["model_dir"])
MODEL_PATH = os.path.join(MODEL_DIR, config["paths"]["model_filename"])
DF_PATH = os.path.join(MODEL_DIR, config["paths"]["df_filename"])