from fastapi import FastAPI
from app.routes import router
from app.logger import setup_logging
from app.recommender import load_pickled_data
from app.config import MODEL_PATH

import asyncio
import os
import logging
from contextlib import asynccontextmanager

setup_logging()

MODEL_REFRESH_INTERVAL = 600
last_model_update_time = 0

async def monitor_model_updates():
    global last_model_update_time
    while True:
        try:
            if os.path.exists(MODEL_PATH):
                model_mtime = os.path.getmtime(MODEL_PATH)
                if model_mtime > last_model_update_time:
                    logging.info("Detected model change. Reloading...")
                    load_pickled_data()
                    last_model_update_time = model_mtime
            else:
                logging.warning("Model path not found for monitoring.")
        except Exception as e:
            logging.error(f"[Model Monitor] Error: {e}")
        await asyncio.sleep(MODEL_REFRESH_INTERVAL)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_pickled_data()
        asyncio.create_task(monitor_model_updates())
        yield
    except Exception as e:
        logging.error(f"Error during app startup: {e}")

app = FastAPI(lifespan=lifespan)
app.include_router(router)
