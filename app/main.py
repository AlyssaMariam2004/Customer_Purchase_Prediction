"""
FastAPI application to serve product recommendations.
This version only loads the pickled model and does NOT schedule retraining.
"""

from fastapi import FastAPI
from app.routes import router
from app.logger import setup_logging

# Setup logging early
setup_logging()

# Initialize FastAPI app
app = FastAPI()
app.include_router(router)

