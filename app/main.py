"""
FastAPI Application for Serving Customer Recommendations

This module initializes a FastAPI application that:
- Loads and monitors a pickled recommendation model.
- Serves a static HTML/CSS/JS frontend.
- Exposes an API endpoint to return product recommendations.
"""

import os
import asyncio
import logging
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

# Internal modules
from app.api.routes import router  # Contains the recommendation API route
from app.core.logger import setup_logging  # Configures structured logging
from app.services.recommender import load_pickled_data  # Loads the model
from app.core.config import MODEL_FILE_PATH  # Path to the model from config.ini

# Setup logging configuration from app/core/logger.py
setup_logging()

# === Constants ===
MODEL_REFRESH_INTERVAL = 600  # Interval (in seconds) to check for model updates
_last_model_update_time = 0  # Last time the model was loaded (epoch timestamp)


async def monitor_model_updates() -> None:
    """
    Background task that monitors the model file for updates.

    If a change is detected (based on modification time), it reloads the model
    using the `load_pickled_data` function. Runs at a fixed interval defined
    by MODEL_REFRESH_INTERVAL.
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
        await asyncio.sleep(MODEL_REFRESH_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that runs on application startup.

    Responsibilities:
    - Loads the initial model using `load_pickled_data`
    - Starts the background monitoring task for model updates
    """
    try:
        load_pickled_data()  # Load model at startup
        asyncio.create_task(monitor_model_updates())  # Start background task
    except Exception as e:
        logging.error(f"[Startup] Failed during application lifespan: {e}")
    yield  # Continue with app startup


# === FastAPI Application ===
app = FastAPI(lifespan=lifespan)

# Mount static files (JS, CSS) served at /static
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Initialize Jinja2 templates located in app/templates/
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Renders the homepage using Jinja2 templates.

    Parameters:
    - request (Request): The incoming HTTP request object

    Returns:
    - HTMLResponse: Rendered index.html with context
    """
    return templates.TemplateResponse("index.html", {"request": request})


# Include API routes (e.g., POST /user for recommendations)
app.include_router(router)



