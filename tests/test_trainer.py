from unittest.mock import patch
import pandas as pd
from app.trainer import maybe_retrain_model

@patch("app.trainer.retrain_model")
@patch("app.trainer.time.time", return_value=9999999999)
def test_maybe_retrain_triggers(mock_time, mock_retrain):
    import app.trainer
    app.trainer.last_retrain_time = 0
    app.trainer.last_row_count = 0

    dummy = pd.DataFrame({
        "Customer ID": ["C1"] * 15,
        "Product ID": ["P1"] * 15,
        "Quantity": [1] * 15,
        "Customer Age": [25] * 15,
        "Customer Gender": ["Male"] * 15,
        "Warehouse ID": ["W1"] * 15
    })

    with patch("pandas.read_csv", return_value=dummy):
        maybe_retrain_model()
        mock_retrain.assert_called_once()
