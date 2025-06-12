from fastapi import FastAPI
from app.routes import router
from app.logger import setup_logging
from app.recommender import load_pickled_data
from app.config import MODEL_PATH

import asyncio
import os
from contextlib import asynccontextmanager

setup_logging()

MODEL_REFRESH_INTERVAL = 600
last_model_update_time = 0

async def monitor_model_updates():
    global last_model_update_time
    while True:
        try:
            model_mtime = os.path.getmtime(MODEL_PATH)
            if model_mtime > last_model_update_time:
                print("[Monitor] New model detected. Reloading...")
                load_pickled_data()
                last_model_update_time = model_mtime
        except Exception as e:
            print(f"[Monitor Error] {e}")
        await asyncio.sleep(MODEL_REFRESH_INTERVAL)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_pickled_data()
    asyncio.create_task(monitor_model_updates())
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(router)

