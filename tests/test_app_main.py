# tests/test_app_main.py

import logging
from unittest.mock import Mock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.main import lifespan


def test_lifespan_model_loaded_successfully():
    """
    Positive Test:
    Ensure load_pickled_data is called when the app starts using lifespan.
    """
    with patch("app.main.load_pickled_data", new_callable=Mock) as mock_load, \
         patch("app.main.monitor_model_updates", new=lambda: None):

        app = FastAPI(lifespan=lifespan)

        @app.get("/")
        async def read_root():
            return {"status": "ok"}

        with TestClient(app) as client:
            response = client.get("/")
            assert response.status_code == 200

        assert mock_load.called is True


def test_lifespan_model_load_failure_logs_error(caplog):
    """
    Negative Test:
    Simulate failure in load_pickled_data and check that the error is logged.
    """
    with patch("app.main.load_pickled_data", side_effect=Exception("Mocked failure")), \
         patch("app.main.monitor_model_updates", new=lambda: None):

        caplog.set_level(logging.ERROR)

        app = FastAPI(lifespan=lifespan)

        @app.get("/")
        async def read_root():
            return {"status": "ok"}

        with TestClient(app) as client:
            response = client.get("/")
            assert response.status_code == 200

        assert "Mocked failure" in caplog.text
        assert "[Startup] Failed during application lifespan" in caplog.text

