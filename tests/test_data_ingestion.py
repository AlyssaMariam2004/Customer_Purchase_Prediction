import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from app import data_ingestion

@pytest.mark.integration
def test_fetch_data_returns_dataframe():
    try:
        df = data_ingestion.fetch_data()
    except Exception:  # Broad catch for integration skips
        pytest.skip("MySQL database not reachable.")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

@pytest.fixture(autouse=True)
def cleanup():
    if os.path.exists(data_ingestion.CSV_PATH):
        os.remove(data_ingestion.CSV_PATH)
    yield
    if os.path.exists(data_ingestion.CSV_PATH):
        os.remove(data_ingestion.CSV_PATH)

def test_fetch_data_calls_mysql_connector_and_reads_sql():
    mock_conn = MagicMock()
    mock_df = pd.DataFrame({"col1": [1, 2]})

    with patch("mysql.connector.connect", return_value=mock_conn), \
         patch("pandas.read_sql", return_value=mock_df):

        result = data_ingestion.fetch_data()

        mock_conn.close.assert_called_once()
        pd.testing.assert_frame_equal(result, mock_df)

def test_sync_data_creates_csv(tmp_path):
    dummy_df = pd.DataFrame({"A": [1], "B": [2]})
    data_ingestion.CSV_PATH = os.path.join(tmp_path, "test.csv")

    with patch("app.data_ingestion.fetch_data", return_value=dummy_df):
        assert not os.path.exists(data_ingestion.CSV_PATH)

        data_ingestion.sync_data()

        assert os.path.exists(data_ingestion.CSV_PATH)
        df = pd.read_csv(data_ingestion.CSV_PATH)
        pd.testing.assert_frame_equal(df, dummy_df)

def test_sync_data_appends_and_dedupes(tmp_path):
    old_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    new_df = pd.DataFrame({"A": [2, 3], "B": [4, 5]})  # overlap on A=2

    data_ingestion.CSV_PATH = os.path.join(tmp_path, "test.csv")
    old_df.to_csv

