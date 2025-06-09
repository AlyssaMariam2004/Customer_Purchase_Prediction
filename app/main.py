"""
Main FastAPI application entry point with background scheduling.

This script initializes the FastAPI app, sets up logging, and manages the app's
lifespan through the async context manager. It performs:

- Initial data sync and model retraining if a CSV already exists.
- Background scheduling of data sync and conditional retraining every 2 minutes.

The scheduler ensures that fresh data from the MySQL database is periodically
fetched and merged into the CSV, and that the model is retrained based on time
and data growth thresholds.
"""

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import router
from app.logger import setup_logging
from app.config import CSV_PATH
from apscheduler.schedulers.background import BackgroundScheduler
import os
import logging

# Initialize logging early, before app creation and any logs
setup_logging()

# Background scheduler that runs independently of the request/response cycle
scheduler = BackgroundScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI app.

    This function handles tasks that should run when the app starts and stops.

    On startup:
    - Schedules periodic jobs to sync data from the database and retrain the model.
    - Runs an initial sync and retrain if the CSV file exists.

    On shutdown:
    - Gracefully stops the background scheduler.

    Args:
        app (FastAPI): The FastAPI application instance.
    """
    from app.data_ingestion import sync_data
    from app.trainer import retrain_model, maybe_retrain_model

    # Start periodic sync and retrain jobs
    scheduler.add_job(sync_data, 'interval', minutes=10, id='sync_data')
    scheduler.add_job(maybe_retrain_model, 'interval', minutes=10, id='maybe_retrain_model')
    scheduler.start()
    logging.info("Background scheduler started.")

    # Initial data load and retraining
    if os.path.exists(CSV_PATH):
        logging.info("Found existing CSV. Running initial sync and retrain.")
        sync_data()
        retrain_model()
    else:
        logging.warning(f"CSV not found at {CSV_PATH}. Skipping initial sync and retrain.")

    yield  # Control is handed over to the FastAPI server

    # Stop scheduled jobs when app shuts down
    scheduler.shutdown()
    logging.info("Scheduler shut down.")

app = FastAPI(lifespan=lifespan)
app.include_router(router)
