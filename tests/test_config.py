from app.config import DB_CONFIG, CSV_PATH, LOG_FILE,BASE_DIR

#Check DB_CONFIG is valid
def test_db_config():
    assert isinstance(DB_CONFIG, dict)
    assert all(k in DB_CONFIG for k in ["host", "user", "password", "database"])

#Check CSV_PATH
def test_csv_path():
    assert CSV_PATH.endswith(".csv")

#Check LOG_FILE
def test_log_file_path():
    assert LOG_FILE.endswith(".log")

#Check if BASE_DIR is a directory
def test_base_dir_is_directory():    
    import os
    assert os.path.isdir(BASE_DIR)
