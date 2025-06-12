"""
test_app_main.py

Unit tests for `app.main`, the entry point for the FastAPI application.
These tests verify:
- Lifespan startup tasks (e.g., model loading and background task start)
- Model monitoring behavior on file change detection
"""

import os
import asyncio
import pytest
from unittest.mock import patch, MagicMock

from fastapi import FastAPI
from app.main import lifespan, monitor_model_updates, MODEL_REFRESH_INTERVAL


@pytest.mark.asyncio
async def test_lifespan_triggers_model_load_and_monitor(monkeypatch):
    """
    Test that the lifespan context manager:
    - Calls the model loader during startup.
    - Starts the model monitoring background task.
    """

    load_mock = MagicMock()
    create_task_mock = MagicMock()

    # Patch model loader and background task scheduler
    monkeypatch.setattr("app.main.load_pickled_data", load_mock)
    monkeypatch.setattr("app.main.asyncio.create_task", create_task_mock)

    app = FastAPI()

    # Trigger lifespan context
    async with lifespan(app):
        pass

    load_mock.assert_called_once()
    create_task_mock.assert_called_once()


@pytest.mark.asyncio
async def test_monitor_model_updates_triggers_reload_on_change(monkeypatch):
    """
    Simulate model file update and test that monitor_model_updates()
    calls load_pickled_data() when file modification time increases.
    """

    # Setup mocks
    test_model_path = "tests/mock_model.pkl"
    with open(test_model_path, "w") as f:
        f.write("mock")

    test_time = os.path.getmtime(test_model_path)

    monkeypatch.setattr("app.main.MODEL_FILE_PATH", test_model_path)
    monkeypatch.setattr("app.main._last_model_update_time", test_time - 100)

    load_mock = MagicMock()
    monkeypatch.setattr("app.main.load_pickled_data", load_mock)

    # Monkeypatch sleep to break loop quickly
    async def dummy_sleep(_):
        raise asyncio.CancelledError()

    monkeypatch.setattr("app.main.asyncio.sleep", dummy_sleep)

    with pytest.raises(asyncio.CancelledError):
        await monitor_model_updates()

    load_mock.assert_called_once()

    # Cleanup
    os.remove(test_model_path)


@pytest.mark.asyncio
async def test_monitor_model_updates_no_model_file(monkeypatch):
    """
    Test monitor_model_updates gracefully handles missing model file.
    """

    monkeypatch.setattr("app.main.MODEL_FILE_PATH", "nonexistent.pkl")
    monkeypatch.setattr("app.main._last_model_update_time", 0)

    load_mock = MagicMock()
    monkeypatch.setattr("app.main.load_pickled_data", load_mock)

    # Monkeypatch sleep to cancel loop early
    async def dummy_sleep(_):
        raise asyncio.CancelledError()

    monkeypatch.setattr("app.main.asyncio.sleep", dummy_sleep)

    with pytest.raises(asyncio.CancelledError):
        await monitor_model_updates()

    load_mock.assert_not_called()
