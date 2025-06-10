"""
Scheduler entry point.
Runs periodic retraining and syncing jobs independent of API.
"""

import logging
from datetime import datetime, timedelta
from apscheduler.schedulers.blocking import BlockingScheduler
from scheduler.trainer import maybe_retrain_model
from scheduler.data_ingestion import sync_data
from app.recommender import load_pickled_data
from app.logger import setup_logging

# Setup logging
setup_logging()



# Setup and start scheduler
scheduler = BlockingScheduler()

now = datetime.now()

# Sync data every 2 minutes, starting immediately
scheduler.add_job(sync_data, 'interval', minutes=2, next_run_time=now)

# Check retrain every 2 minutes, starting 5 seconds after sync
scheduler.add_job(maybe_retrain_model, 'interval', minutes=2, next_run_time=now + timedelta(seconds=5))

if __name__ == "__main__":
    logging.info("Scheduler started.")
    sync_data()  # optional: pull data immediately
    maybe_retrain_model()  # optional: perform initial check
    scheduler.start()
