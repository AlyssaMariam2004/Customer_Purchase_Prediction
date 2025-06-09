import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from app import data_ingestion

class TestDataIngestion:
    """
    Test suite for the data_ingestion module.

    Tests both integration and unit aspects:
    - Integration test for fetch_data() connecting to MySQL and returning data.
    - Unit tests for fetch_data() using mocks.
    - sync_data() functionality ensuring correct CSV creation, appending, and deduplication.
    
    Uses pytest's tmp_path fixture to isolate test CSV files in temporary directories,
    preventing any modification or deletion of the actual production CSV file.
    """

    @pytest.mark.integration
    def test_fetch_data_returns_dataframe(self):
        """
        Integration test for fetch_data().

        Verifies fetch_data() connects to the MySQL database and returns a
        non-empty pandas DataFrame. Skips test if the database is unreachable.
        """
        try:
            df = data_ingestion.fetch_data()
        except Exception:
            pytest.skip("MySQL database not reachable.")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_fetch_data_calls_mysql_connector_and_reads_sql(self):
        """
        Unit test for fetch_data() with mocks.

        Mocks mysql.connector.connect and pandas.read_sql to simulate database
        connection and query. Verifies connection close and correct DataFrame return.
        """
        mock_conn = MagicMock()
        mock_df = pd.DataFrame({"col1": [1, 2]})

        with patch("mysql.connector.connect", return_value=mock_conn), \
             patch("pandas.read_sql", return_value=mock_df):

            result = data_ingestion.fetch_data()

            mock_conn.close.assert_called_once()
            pd.testing.assert_frame_equal(result, mock_df)

    def test_sync_data_creates_csv(self, tmp_path):
        """
        Test sync_data() creates a CSV file correctly in a temporary path.

        Mocks fetch_data() to return a dummy DataFrame,
        runs sync_data(), and verifies that the CSV file is created
        and its content matches the dummy DataFrame.
        """
        dummy_df = pd.DataFrame({"A": [1], "B": [2]})

        # Override CSV_PATH to a temp test file inside tmp_path
        data_ingestion.CSV_PATH = os.path.join(tmp_path, "test.csv")

        with patch("app.data_ingestion.fetch_data", return_value=dummy_df):
            # Ensure the CSV does not exist before sync
            assert not os.path.exists(data_ingestion.CSV_PATH)

            data_ingestion.sync_data()

            # Check the CSV was created
            assert os.path.exists(data_ingestion.CSV_PATH)

            # Read back and check contents match dummy_df exactly
            df = pd.read_csv(data_ingestion.CSV_PATH)
            pd.testing.assert_frame_equal(df, dummy_df)

    def test_sync_data_appends_and_dedupes(self, tmp_path):
        """
        Test sync_data() appends new data to existing CSV and removes duplicates.

        Creates an initial CSV file with old data, mocks fetch_data() to return new data
        with some overlapping rows, runs sync_data(), then verifies
        the final CSV contains all unique rows combined.
        """
        old_df = pd.DataFrame({
            "Order ID": [1, 2],
            "Product ID": [101, 102],
            "SKU ID": [1001, 1002],
            "Quantity": [3, 4]
        })

        new_df = pd.DataFrame({
            "Order ID": [2, 3],
            "Product ID": [102, 103],
            "SKU ID": [1002, 1003],
            "Quantity": [4, 5]
        })  # Note overlap on Order ID = 2

        # Save old_df to temp CSV file
        data_ingestion.CSV_PATH = os.path.join(tmp_path, "test.csv")
        old_df.to_csv(data_ingestion.CSV_PATH, index=False)

        with patch("app.data_ingestion.fetch_data", return_value=new_df):
            data_ingestion.sync_data()

            combined_df = pd.read_csv(data_ingestion.CSV_PATH)

            expected_df = pd.DataFrame({
                "Order ID": [1, 2, 3],
                "Product ID": [101, 102, 103],
                "SKU ID": [1001, 1002, 1003],
                "Quantity": [3, 4, 5]
            })

            # Compare ignoring row order and index
            pd.testing.assert_frame_equal(
                combined_df.sort_values(by="Order ID").reset_index(drop=True),
                expected_df.sort_values(by="Order ID").reset_index(drop=True),
            )
