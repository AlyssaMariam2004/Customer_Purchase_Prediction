from unittest.mock import patch
import pandas as pd
from app.trainer import maybe_retrain_model

@patch("app.trainer.retrain_model")
@patch("app.trainer.time.time")
def test_maybe_retrain_behavior(mock_time, mock_retrain):
    import app.trainer

    # --- Case 1: retrain should trigger ---
    # Set last retrain time far in the past and last_row_count low
    app.trainer.last_retrain_time = 0
    app.trainer.last_row_count = 0

    # Mock current time large enough to pass threshold
    mock_time.return_value = 9999999999

    dummy_data_trigger = pd.DataFrame({
        "Customer ID": ["C1"] * 15,
        "Product ID": ["P1"] * 15,
        "Quantity": [1] * 15,
        "Customer Age": [25] * 15,
        "Customer Gender": ["Male"] * 15,
        "Warehouse ID": ["W1"] * 15
    })

    with patch("pandas.read_csv", return_value=dummy_data_trigger):
        maybe_retrain_model()
        mock_retrain.assert_called_once()

    mock_retrain.reset_mock()  # reset mock for next case

    # --- Case 2: retrain should NOT trigger ---
    # Set last retrain time recent, so threshold not reached
    app.trainer.last_retrain_time = 9999999990  # 9 seconds before mocked time
    app.trainer.last_row_count = 14  # Almost same row count

    dummy_data_no_trigger = pd.DataFrame({
        "Customer ID": ["C1"] * 15,
        "Product ID": ["P1"] * 15,
        "Quantity": [1] * 15,
        "Customer Age": [25] * 15,
        "Customer Gender": ["Male"] * 15,
        "Warehouse ID": ["W1"] * 15
    })

    with patch("pandas.read_csv", return_value=dummy_data_no_trigger):
        maybe_retrain_model()
        mock_retrain.assert_not_called()

