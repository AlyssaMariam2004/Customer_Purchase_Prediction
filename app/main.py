from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import router
from app.logger import setup_logging
from app.config import CSV_PATH
import os
import logging

@asynccontextmanager #used by FastAPI to manage app startup and shutdown events
async def lifespan(app: FastAPI): #The lifespan function runs once before the app starts serving requests, and once on shutdown
    from app.data_ingestion import sync_data
    from app.trainer import retrain_model

    if os.path.exists(CSV_PATH): #looks for the csv file and if it exists runs the two functions
        logging.info("Found existing CSV. Running sync and possible retrain.")
        sync_data()
        retrain_model()
    else:
        logging.warning(f"CSV not found at {CSV_PATH}. Skipping initial sync and retrain.")

    yield  # Runs the app

app = FastAPI(lifespan=lifespan)#creating fastapi instance

#initiates logging
setup_logging()

#defines the api routes
app.include_router(router)


