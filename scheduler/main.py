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

from scheduler.trainer import maybe_retrain_model
from scheduler.data_ingestion import sync_data
from scheduler.logger import setup_logging

# Configure global logging
setup_logging()

# Create an instance of the scheduler
scheduler = BlockingScheduler()


def start_scheduler():
    """
    Initializes and starts the APScheduler with sync and retrain jobs.
    Logs any exception during setup or execution.
    """
    now = datetime.now()

    try:
        # Schedule `sync_data` to run every 2 minutes
        scheduler.add_job(
            sync_data,
            trigger='interval',
            minutes=2,
            next_run_time=now
        )

        # Schedule `maybe_retrain_model` to run every 2 minutes (offset)
        scheduler.add_job(
            maybe_retrain_model,
            trigger='interval',
            minutes=2,
            next_run_time=now + timedelta(seconds=5)
        )

        logging.info("Scheduler starting...")

        # Run initial executions before interval triggers
        sync_data()
        maybe_retrain_model()

        # Start the scheduler loop
        scheduler.start()

    except Exception as e:
        logging.error(f"Error in scheduler initialization: {e}")


# Run only if this script is the entry point
if __name__ == "__main__":
    start_scheduler()
