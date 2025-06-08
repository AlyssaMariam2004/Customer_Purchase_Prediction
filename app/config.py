import os

# Get project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#Database Configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "22mid0076",
    "database": "SalesDB"
}

CSV_PATH = os.path.join(BASE_DIR, "data", "NewData1.csv")

#Scheduling model retraining
RETRAIN_INTERVAL = 48 * 60 * 60  # 48 hours
ROW_GROWTH_THRESHOLD = 1000 #number of new enteries 

#Logging Setup
LOG_FILE = os.path.join(BASE_DIR, "logs", "recommendation_system.log")
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
