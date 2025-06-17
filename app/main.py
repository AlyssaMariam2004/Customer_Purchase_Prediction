import os
import asyncio
import logging
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

from app.api.routes import router
from app.core.logger import setup_logging
from app.services.recommender import load_pickled_data
from app.core.config import MODEL_FILE_PATH

setup_logging()

MODEL_REFRESH_INTERVAL = 600
_last_model_update_time = 0

async def monitor_model_updates() -> None:
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
    try:
        load_pickled_data()
        asyncio.create_task(monitor_model_updates())
    except Exception as e:
        logging.error(f"[Startup] Failed during application lifespan: {e}")
    yield

app = FastAPI(lifespan=lifespan)

# Mount static and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Route for home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API routes
app.include_router(router)



