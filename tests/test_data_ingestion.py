# tests/test_data_ingestion.py

import pytest
import pandas as pd
from unittest import mock
from unittest.mock import MagicMock, patch
from scheduler import data_ingestion



@patch("scheduler.data_ingestion.mysql.connector.connect")
@patch("pandas.read_sql")
def test_fetch_data_success(mock_read_sql, mock_connect):
    """
    Positive Test:
    Should return a non-empty DataFrame when DB connection and query succeed.
    """
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    mock_read_sql.return_value = pd.DataFrame({"col1": [1], "col2": [2]})

    df = data_ingestion.fetch_data()
    assert not df.empty
    mock_connect.assert_called_once()
    mock_read_sql.assert_called_once()


@patch("scheduler.data_ingestion.mysql.connector.connect", side_effect=Exception("DB Error"))
def test_fetch_data_connection_failure(mock_connect):
    """
    Negative Test:
    Should return empty DataFrame when DB connection fails.
    """
    df = data_ingestion.fetch_data()
    assert df.empty


@patch("scheduler.data_ingestion.fetch_data")
@patch("pandas.read_csv")
@patch("pandas.DataFrame.to_csv")
@patch("os.path.exists")
def test_sync_data_with_existing_csv(mock_exists, mock_to_csv, mock_read_csv, mock_fetch_data):
    """
    Positive Test:
    Should merge fetched data with existing CSV and call to_csv.
    """
    mock_exists.return_value = True
    mock_fetch_data.return_value = pd.DataFrame({"A": [1]})
    mock_read_csv.return_value = pd.DataFrame({"A": [1]})  # duplicate to test deduplication

    data_ingestion.sync_data()
    mock_to_csv.assert_called_once()


@patch("scheduler.data_ingestion.fetch_data", return_value=pd.DataFrame())
def test_sync_data_with_empty_fetch(mock_fetch_data):
    """
    Negative Test:
    Should skip writing if fetch_data returns empty DataFrame.
    """
    with patch("pandas.DataFrame.to_csv") as mock_to_csv:
        data_ingestion.sync_data()
        mock_to_csv.assert_not_called()


@patch("scheduler.data_ingestion.fetch_data", return_value=pd.DataFrame({"A": [1]}))
@patch("os.path.exists", return_value=False)
@patch("pandas.DataFrame.to_csv", side_effect=Exception("Write error"))
def test_sync_data_csv_write_error(mock_to_csv, mock_exists, mock_fetch_data):
    """
    Negative Test:
    Should log error if writing CSV fails.
    """
    with patch("logging.error") as mock_log:
        data_ingestion.sync_data()
        mock_log.assert_called_with("Error during CSV write in sync_data: Write error")
