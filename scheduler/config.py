import os
from configparser import RawConfigParser

config = RawConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))

def get_path(section, key):
    return os.path.abspath(config.get(section, key))

# Paths
CSV_PATH = get_path("paths", "csv_filename")
MODEL_DIR = get_path("paths", "model_dir")
MODEL_PATH = os.path.join(MODEL_DIR, config.get("paths", "model_filename"))
DF_PATH = os.path.join(MODEL_DIR, config.get("paths", "df_filename"))

# Database
DB_CONFIG = {
    'host': config.get("database", "host"),
    'user': config.get("database", "user"),
    'password': config.get("database", "password"),
    'database': config.get("database", "database")
}

# Logging
LOG_FILE = config.get("logging", "log_file")
LOG_LEVEL = config.get("logging", "log_level")
LOG_FORMAT = config.get("logging", "log_format")

# Retrain Parameters
RETRAIN_INTERVAL = int(config.get("retraining", "interval_seconds"))
ROW_GROWTH_THRESHOLD = int(config.get("retraining", "row_growth_threshold"))
