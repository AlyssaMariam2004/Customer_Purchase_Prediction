"""
main.py

This module initializes and runs scheduled tasks using APScheduler.
It handles periodic:
- Data synchronization from the database to CSV.
- Model retraining based on configured conditions.

Logging is initialized at startup.
"""

import logging
from datetime import datetime, timedelta
from apscheduler.schedulers.blocking import BlockingScheduler

from trainer import maybe_retrain_model
from data_ingestion import sync_data
from logger import setup_logging

# Configure global logging
setup_logging()

# Create an instance of the scheduler
scheduler = BlockingScheduler()

# Get current timestamp to schedule initial runs
now = datetime.now()

try:
    # Schedule `sync_data` to run every 2 minutes
    scheduler.add_job(
        sync_data,
        trigger='interval',
        minutes=2,
        next_run_time=now  # first run immediately
    )

    # Schedule `maybe_retrain_model` to run every 2 minutes (with slight offset)
    scheduler.add_job(
        maybe_retrain_model,
        trigger='interval',
        minutes=2,
        next_run_time=now + timedelta(seconds=5)  # first run after 5 seconds
    )

    if __name__ == "__main__":
        logging.info("Scheduler starting...")
        
        # Initial manual run before interval jobs
        sync_data()
        maybe_retrain_model()

        # Start the scheduler loop
        scheduler.start()

except Exception as e:
    logging.error(f"Error in scheduler initialization: {e}")
