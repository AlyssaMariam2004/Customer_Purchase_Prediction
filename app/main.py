from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import router
from app.logger import setup_logging
from app.config import CSV_PATH
import os
import logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.data_ingestion import sync_data
    from app.trainer import retrain_model

    if os.path.exists(CSV_PATH):
        logging.info("Found existing CSV. Running sync and possible retrain.")
        sync_data()
        retrain_model()
    else:
        logging.warning(f"CSV not found at {CSV_PATH}. Skipping initial sync and retrain.")

    yield  # Run the app

app = FastAPI(lifespan=lifespan)

setup_logging()
app.include_router(router)


