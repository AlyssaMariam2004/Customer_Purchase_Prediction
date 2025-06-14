"""
main.py

This is the entry point for the FastAPI application.
It sets up logging, loads the initial machine learning model, 
monitors for model file changes, and mounts all application routes.
"""

import os
import asyncio
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routes import router
from app.core.logger import setup_logging
from app.services.recommender import load_pickled_data
from app.core.config import MODEL_FILE_PATH

# Set up logging configuration
setup_logging()

# Interval in seconds to check for model file changes
MODEL_REFRESH_INTERVAL = 600  # 10 minutes
_last_model_update_time = 0  # Internal tracker for last model update timestamp


async def monitor_model_updates() -> None:
    """
    Background task to monitor changes to the model file.
    Reloads the model if the file has been updated.
    """
    global _last_model_update_time
    while True:
        try:
            if os.path.exists(MODEL_FILE_PATH):
                model_mtime = os.path.getmtime(MODEL_FILE_PATH)
                if model_mtime > _last_model_update_time:
                    logging.info("Detected model change. Reloading model...")
                    load_pickled_data()
                    _last_model_update_time = model_mtime
            else:
                logging.warning("Model file path does not exist.")
        except Exception as e:
            logging.error(f"[Model Monitor] Unexpected error: {e}")
        
        # Wait before checking again
        await asyncio.sleep(MODEL_REFRESH_INTERVAL)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles the application's startup and shutdown logic.

    - Loads the model at startup
    - Starts the background task to monitor model changes
    """
    try:
        load_pickled_data()
        asyncio.create_task(monitor_model_updates())
    except Exception as e:
        logging.error(f"[Startup] Failed during application lifespan: {e}")

    yield  # Required for FastAPI lifespan to work



# Initialize FastAPI app with custom lifespan handler
app = FastAPI(lifespan=lifespan)

# Register API routes
app.include_router(router)



