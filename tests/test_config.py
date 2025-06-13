# tests/test_config.py

import os
import tempfile
import configparser
import pytest
from importlib import reload
from unittest import mock


@pytest.fixture
def temp_config_path():
    """
    Fixture to create a temporary config.ini file and return its path.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "config.ini")

        config = configparser.ConfigParser()
        config["paths"] = {
            "csv_path": "./data/data.csv",
            "model_dir": "./models",
            "model_path": "./models/model.pkl",
            "df_path": "./data/final_df.pkl"
        }
        config["database"] = {
            "host": "localhost",
            "user": "user",
            "password": "pass",
            "database": "mydb"
        }
        config["logging"] = {
            "log_file": "logs/app.log",
            "log_level": "INFO",
            "log_format": "%(levelname)s:%(message)s"
        }
        config["retrain"] = {
            "interval": "60",
            "row_growth_threshold": "500"
        }
        config["model"] = {
            "top_n": "5"
        }

        with open(config_path, "w") as f:
            config.write(f)

        yield config_path


def test_get_absolute_path_success(temp_config_path):
    """
    Positive Test:
    Should return an absolute path for a valid section and key using mocked config.
    """
    with mock.patch("app.config.os.path.join", return_value=temp_config_path):
        import app.config as config_loader
        reload(config_loader)
        path = config_loader.get_absolute_path("paths", "csv_path")
        assert isinstance(path, str)
        assert os.path.isabs(path)


def test_get_absolute_path_invalid_key(temp_config_path):
    """
    Negative Test:
    Should raise NoOptionError for an invalid key.
    """
    with mock.patch("app.config.os.path.join", return_value=temp_config_path):
        import app.config as config_loader
        reload(config_loader)
        with pytest.raises(configparser.NoOptionError):
            config_loader.get_absolute_path("paths", "invalid_key")


def test_get_absolute_path_invalid_section(temp_config_path):
    """
    Negative Test:
    Should raise NoSectionError for an invalid section.
    """
    with mock.patch("app.config.os.path.join", return_value=temp_config_path):
        import app.config as config_loader
        reload(config_loader)
        with pytest.raises(configparser.NoSectionError):
            config_loader.get_absolute_path("invalid_section", "csv_path")

