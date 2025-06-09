from configparser import RawConfigParser
import os

# Setup config parser
config = RawConfigParser()
config.read(os.path.join(os.path.dirname(__file__), "config.ini"))

# Project base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Database Configuration
DB_CONFIG = {
    "host": config["database"]["host"],
    "user": config["database"]["user"],
    "password": config["database"]["password"],
    "database": config["database"]["database"]
}

# Dataset CSV path
CSV_PATH = os.path.join(BASE_DIR, "data", config["paths"]["csv_filename"])

# Retraining parameters
RETRAIN_INTERVAL = int(config["retraining"]["interval_seconds"])
ROW_GROWTH_THRESHOLD = int(config["retraining"]["row_growth_threshold"])

# Logging
LOG_FILE = os.path.join(BASE_DIR, config["logging"]["log_file"])
LOG_LEVEL = config["logging"]["log_level"]
LOG_FORMAT = config["logging"]["log_format"]  
