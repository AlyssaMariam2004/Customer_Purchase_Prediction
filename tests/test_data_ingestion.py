"""
test_data_ingestion.py

Unit tests for the data_ingestion module which handles MySQL data fetching
and CSV synchronization for offline training and predictions.
"""

import os
import pytest
import pandas as pd
from unittest import mock
from unittest.mock import MagicMock, patch

import scheduler.data_ingestion as ingestion


@pytest.fixture
def mock_db_dataframe():
    return pd.DataFrame({
        "Order ID": [1],
        "Customer ID": ["CUST1"],
        "Warehouse ID": [101],
        "Customer Age": [25],
        "Customer Gender": ["F"],
        "Date": ["2023-01-01"],
        "Product ID": ["PROD1"],
        "SKU ID": ["SKU123"],
        "Category": ["Electronics"],
        "Quantity": [2],
        "Price per Unit": [199.99]
    })


@patch("scheduler.data_ingestion.mysql.connector.connect")
@patch("pandas.read_sql")
def test_fetch_data_success(mock_read_sql, mock_connect, mock_db_dataframe):
    """
    Test fetch_data returns data correctly when MySQL query succeeds.
    """
    mock_read_sql.return_value = mock_db_dataframe
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn

    result_df = ingestion.fetch_data()
    assert isinstance(result_df, pd.DataFrame)
    assert not result_df.empty
    assert "Customer ID" in result_df.columns

    mock_connect.assert_called_once()
    mock_conn.close.assert_called_once()


@patch("scheduler.data_ingestion.mysql.connector.connect", side_effect=Exception("DB failure"))
def test_fetch_data_db_connection_failure(mock_connect):
    """
    Test fetch_data handles DB connection error and returns empty DataFrame.
    """
    result_df = ingestion.fetch_data()
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.empty


@patch("scheduler.data_ingestion.fetch_data")
def test_sync_data_new_csv_created(mock_fetch_data, tmp_path, mock_db_dataframe):
    """
    Test sync_data creates a new CSV if one does not already exist.
    """
    # Simulate data fetch
    mock_fetch_data.return_value = mock_db_dataframe

    # Replace global CSV_PATH with temp file path
    test_csv_path = tmp_path / "test_data.csv"
    ingestion.CSV_PATH = str(test_csv_path)

    ingestion.sync_data()
    assert test_csv_path.exists()
    
    df_written = pd.read_csv(test_csv_path)
    assert not df_written.empty
    assert "Product ID" in df_written.columns


@patch("scheduler.data_ingestion.fetch_data")
def test_sync_data_existing_csv_merge(mock_fetch_data, tmp_path, mock_db_dataframe):
    """
    Test sync_data merges new data with existing CSV and deduplicates.
    """
    # Simulate data fetch
    mock_fetch_data.return_value = mock_db_dataframe

    # Create existing CSV with the same data
    test_csv_path = tmp_path / "existing.csv"
    mock_db_dataframe.to_csv(test_csv_path, index=False)
    ingestion.CSV_PATH = str(test_csv_path)

    ingestion.sync_data()
    
    df_written = pd.read_csv(test_csv_path)
    assert len(df_written) == 1  # Deduplication ensures no duplicates


@patch("scheduler.data_ingestion.fetch_data", return_value=pd.DataFrame())
def test_sync_data_no_new_data_logged(mock_fetch_data, caplog):
    """
    Test sync_data logs a warning when no new data is fetched.
    """
    ingestion.sync_data()
    assert "No data fetched from the database." in caplog.text
