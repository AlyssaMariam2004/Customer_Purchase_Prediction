import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from app.services import recommender


def test_load_pickled_data_successful(tmp_path):
    """
    Positive Test:
    Ensure that pickled data and model files are loaded successfully into global variables.
    """
    df_sample = pd.DataFrame({"Customer ID": ["C1"], "Product ID": ["P1"], "Quantity": [1], "Cluster": [0]})
    model_sample = pd.DataFrame({"Cluster": [0]}, index=["C1"])

    data_path = tmp_path / "df.pkl"
    model_path = tmp_path / "model.pkl"

    # Save sample DataFrames
    df_sample.to_pickle(data_path)
    model_sample.to_pickle(model_path)

    # Patch config paths
    with patch("app.recommender.DATAFRAME_PATH", str(data_path)), \
         patch("app.recommender.MODEL_FILE_PATH", str(model_path)):
        recommender.load_pickled_data()
        assert isinstance(recommender.df, pd.DataFrame)
        assert isinstance(recommender.final_df, pd.DataFrame)


def test_load_pickled_data_file_missing():
    """
    Negative Test:
    Raise RuntimeError if pickled files do not exist.
    """
    with patch("app.recommender.DATAFRAME_PATH", "nonexistent/df.pkl"), \
         patch("app.recommender.MODEL_FILE_PATH", "nonexistent/model.pkl"):
        with pytest.raises(RuntimeError) as exc_info:
            recommender.load_pickled_data()
        assert "Pickled model or data file not found" in str(exc_info.value)


def test_recommend_products_successful():
    """
    Positive Test:
    Recommend products for a valid customer in a valid cluster.
    """
    # Purchase data
    recommender.df = pd.DataFrame({
        "Customer ID": ["C1", "C2", "C2"],
        "Product ID": ["P1", "P2", "P3"],
        "Quantity": [1, 2, 3],
        "Cluster": [0, 0, 0],
    })

    # Cluster mapping
    recommender.final_df = pd.DataFrame({"Cluster": [0, 0]}, index=["C1", "C2"])

    result = recommender.recommend_products("C1", top_n=2)

    assert isinstance(result, list)
    assert all(isinstance(prod, str) for prod in result)



def test_recommend_products_customer_not_found():
    """
    Negative Test:
    Raise 404 if customer ID is not in the model.
    """
    recommender.df = pd.DataFrame()
    recommender.final_df = pd.DataFrame({"Cluster": [0]}, index=["C2"])

    with pytest.raises(HTTPException) as exc_info:
        recommender.recommend_products("C1")
    assert exc_info.value.status_code == 404
    assert "Customer ID" in exc_info.value.detail


def test_recommend_products_model_not_loaded():
    """
    Negative Test:
    Raise 500 if the model is not loaded.
    """
    recommender.df = None
    recommender.final_df = None

    with pytest.raises(HTTPException) as exc_info:
        recommender.recommend_products("C1")
    assert exc_info.value.status_code == 500
    assert "Model not loaded" in exc_info.value.detail
