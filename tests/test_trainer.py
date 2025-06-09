from unittest.mock import patch
import pandas as pd
from app.trainer import maybe_retrain_model

@patch("app.trainer.retrain_model")
@patch("app.trainer.time.time")
def test_maybe_retrain_behavior(mock_time, mock_retrain):
    """
    Test the behavior of maybe_retrain_model function under different conditions.

    This test verifies that the retraining logic triggers correctly when
    thresholds for time elapsed and new data rows are met, and does not trigger
    when those thresholds are not met.

    Args:
        mock_time (MagicMock): Mock for time.time() to control current time.
        mock_retrain (MagicMock): Mock for retrain_model function to check if called.
    """

    import app.trainer

    # --- Case 1: retrain should trigger ---
    # Setup initial state with last retrain far in the past and low last row count
    app.trainer.last_retrain_time = 0
    app.trainer.last_row_count = 0

    # Mock current time to a large value to simulate time threshold passed
    mock_time.return_value = 9999999999

    # Create dummy data large enough to exceed row count threshold to trigger retrain
    dummy_data_trigger = pd.DataFrame({
        "Customer ID": ["C1"] * 15,
        "Product ID": ["P1"] * 15,
        "Quantity": [1] * 15,
        "Customer Age": [25] * 15,
        "Customer Gender": ["Male"] * 15,
        "Warehouse ID": ["W1"] * 15
    })

    # Patch pandas.read_csv to return the dummy data
    with patch("pandas.read_csv", return_value=dummy_data_trigger):
        maybe_retrain_model()
        # Assert retrain_model was called because thresholds were exceeded
        mock_retrain.assert_called_once()

    # Reset mock before next case
    mock_retrain.reset_mock()

    # --- Case 2: retrain should NOT trigger ---
    # Setup initial state with last retrain time recent (threshold not passed)
    app.trainer.last_retrain_time = 9999999990  # 9 seconds before mocked current time
    app.trainer.last_row_count = 14  # Almost same number of rows as current data

    # Create dummy data same size as previous, so row count threshold not exceeded
    dummy_data_no_trigger = pd.DataFrame({
        "Customer ID": ["C1"] * 15,
        "Product ID": ["P1"] * 15,
        "Quantity": [1] * 15,
        "Customer Age": [25] * 15,
        "Customer Gender": ["Male"] * 15,
        "Warehouse ID": ["W1"] * 15
    })

    # Patch pandas.read_csv again for this data
    with patch("pandas.read_csv", return_value=dummy_data_no_trigger):
        maybe_retrain_model()
        # Assert retrain_model was NOT called because thresholds not exceeded
        mock_retrain.assert_not_called()
