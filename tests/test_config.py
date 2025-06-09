from app.config import DB_CONFIG, CSV_PATH, LOG_FILE, BASE_DIR

def test_db_config():
    """
    Test to verify that the database configuration is valid.

    Checks:
    - DB_CONFIG is a dictionary.
    - Required keys ("host", "user", "password", "database") are present in DB_CONFIG.

    Raises:
        AssertionError: If any check fails.
    """
    assert isinstance(DB_CONFIG, dict), "DB_CONFIG should be a dictionary."
    assert all(k in DB_CONFIG for k in ["host", "user", "password", "database"]), \
        "DB_CONFIG missing one or more required keys: 'host', 'user', 'password', 'database'."

def test_csv_path():
    """
    Test to verify that the CSV_PATH ends with '.csv'.

    Raises:
        AssertionError: If CSV_PATH does not end with '.csv'.
    """
    assert CSV_PATH.endswith(".csv"), f"CSV_PATH should end with '.csv', got {CSV_PATH}"

def test_log_file_path():
    """
    Test to verify that the log file path ends with '.log'.

    Raises:
        AssertionError: If LOG_FILE does not end with '.log'.
    """
    assert LOG_FILE.endswith(".log"), f"LOG_FILE should end with '.log', got {LOG_FILE}"

def test_base_dir_is_directory():
    """
    Test to verify that BASE_DIR is a valid directory on the filesystem.

    Raises:
        AssertionError: If BASE_DIR is not a directory.
    """
    import os
    assert os.path.isdir(BASE_DIR), f"BASE_DIR should be a directory, but '{BASE_DIR}' is not."
