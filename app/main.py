from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import router
from app.logger import setup_logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    from app.data_ingestion import sync_data
    from app.trainer import retrain_model
    sync_data()
    retrain_model()

    yield  # run the app

    # Shutdown code (optional)

app = FastAPI(lifespan=lifespan)

setup_logging()
app.include_router(router)

