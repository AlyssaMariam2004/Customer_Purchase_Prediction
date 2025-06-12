"""
test_config.py

Unit tests for the `app.config` module, which handles application-wide configuration
loading from a `.ini` file. Tests include validation of path resolution, configuration
value types, and expected key presence in configuration dictionaries.
"""

import os
import pytest
from app import config


def test_csv_file_path_is_absolute_and_exists():
    """
    Test if the CSV file path is an absolute string and points to a valid file or directory.
    """
    assert isinstance(config.CSV_FILE_PATH, str)
    assert os.path.isabs(config.CSV_FILE_PATH)


def test_model_directory_is_absolute_and_exists():
    """
    Test if the model directory path is absolute and a valid path.
    """
    assert isinstance(config.MODEL_DIRECTORY, str)
    assert os.path.isabs(config.MODEL_DIRECTORY)


def test_model_file_path_is_absolute():
    """
    Ensure the model file path is an absolute path.
    """
    assert isinstance(config.MODEL_FILE_PATH, str)
    assert os.path.isabs(config.MODEL_FILE_PATH)


def test_dataframe_path_is_absolute():
    """
    Ensure the DataFrame pickle file path is an absolute path.
    """
    assert isinstance(config.DATAFRAME_PATH, str)
    assert os.path.isabs(config.DATAFRAME_PATH)


def test_database_config_contains_expected_keys():
    """
    Test if database config contains required keys and they are non-empty strings.
    """
    required_keys = {'host', 'user', 'password', 'database'}
    assert set(config.DATABASE_CONFIG.keys()) == required_keys

    for key in required_keys:
        assert isinstance(config.DATABASE_CONFIG[key], str)
        assert config.DATABASE_CONFIG[key] != ""


def test_logging_config_values():
    """
    Validate that logging configuration values are strings and log level is one of the valid levels.
    """
    assert isinstance(config.LOGGING_FILE_PATH, str)
    assert isinstance(config.LOGGING_LEVEL, str)
    assert isinstance(config.LOGGING_FORMAT, str)
    assert config.LOGGING_LEVEL.upper() in {
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    }


def test_retrain_config_values_are_integers():
    """
    Ensure retraining config values are integers and positive.
    """
    assert isinstance(config.MODEL_RETRAIN_INTERVAL, int)
    assert config.MODEL_RETRAIN_INTERVAL > 0

    assert isinstance(config.ROW_GROWTH_THRESHOLD, int)
    assert config.ROW_GROWTH_THRESHOLD > 0


def test_default_top_n_is_positive_integer():
    """
    Validate that default top N recommendations is a positive integer.
    """
    assert isinstance(config.DEFAULT_TOP_N, int)
    assert config.DEFAULT_TOP_N > 0

