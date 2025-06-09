"""
Main FastAPI application entry point.

This script sets up the FastAPI app, configures logging, handles startup tasks
such as syncing data and retraining the model, and includes API routes.
"""

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import router
from app.logger import setup_logging
from app.config import CSV_PATH
import os
import logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    This runs once when the app starts and once when it shuts down.
    - On startup, checks if the CSV file exists.
    - If it exists, it syncs the latest data from the database and retrains the model.
    - If not, logs a warning and skips sync and retraining.
    """
    from app.data_ingestion import sync_data
    from app.trainer import retrain_model

    if os.path.exists(CSV_PATH):
        logging.info("Found existing CSV. Running sync and possible retrain.")
        sync_data()
        retrain_model()
    else:
        logging.warning(f"CSV not found at {CSV_PATH}. Skipping initial sync and retrain.")

    yield  # Start serving requests

# Create the FastAPI application instance with lifespan handler
app = FastAPI(lifespan=lifespan)

# Set up file-based logging
setup_logging()

# Register all API routes from the router module
app.include_router(router)
