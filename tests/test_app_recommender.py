'''
test_recommender.py

Unit tests for app.recommender module.
Includes tests for:
- Model loading (load_pickled_data)
- Product recommendations (recommend_products)
'''

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from app import recommender


@pytest.fixture
def mock_data():
    """Fixture to set up mock df and final_df for recommendation tests."""
    df = pd.DataFrame({
        'Customer ID': ['CUST1', 'CUST2', 'CUST2', 'CUST3', 'CUST1'],
        'Product ID': ['P1', 'P1', 'P2', 'P3', 'P4'],
        'Quantity': [1, 1, 2, 3, 1],
        'Cluster': [0, 0, 0, 1, 0]
    })

    final_df = pd.DataFrame({
        'Cluster': [0, 0, 1]
    }, index=['CUST1', 'CUST2', 'CUST3'])

    return df, final_df


@patch("app.recommender.joblib.load")
@patch("app.recommender.os.path.exists")
def test_load_pickled_data_success(mock_exists, mock_load):
    """
    Tests successful loading of model and dataframe pickle files.
    """
    mock_exists.return_value = True
    mock_load.side_effect = [pd.DataFrame(), pd.DataFrame()]  # Return valid DataFrames

    recommender.load_pickled_data()
    assert isinstance(recommender.df, pd.DataFrame)
    assert isinstance(recommender.final_df, pd.DataFrame)


@patch("app.recommender.os.path.exists")
def test_load_pickled_data_file_not_found(mock_exists):
    """
    Tests if FileNotFoundError is handled when pickle files are missing.
    """
    mock_exists.return_value = False
    with pytest.raises(RuntimeError):
        recommender.load_pickled_data()


@patch("app.recommender.df", create=True)
@patch("app.recommender.final_df", create=True)
def test_recommend_products_success(mock_final_df, mock_df, mock_data):
    """
    Tests successful recommendation generation.
    """
    df, final_df = mock_data
    mock_df.__get__ = lambda *_: df
    mock_final_df.__get__ = lambda *_: final_df

    recommender.df = df
    recommender.final_df = final_df

    recommendations = recommender.recommend_products("CUST1", top_n=2)
    assert isinstance(recommendations, list)
    assert len(recommendations) <= 2


def test_recommend_products_model_not_loaded():
    """
    Tests error when recommendation is requested before loading model.
    """
    recommender.df = None
    recommender.final_df = None

    with pytest.raises(HTTPException) as exc_info:
        recommender.recommend_products("CUST1")
    assert exc_info.value.status_code == 500


def test_recommend_products_invalid_customer(mock_data):
    """
    Tests handling of unknown customer ID.
    """
    df, final_df = mock_data
    recommender.df = df
    recommender.final_df = final_df

    with pytest.raises(HTTPException) as exc_info:
        recommender.recommend_products("INVALID")
    assert exc_info.value.status_code == 404


def test_recommend_products_no_purchase_history(mock_data):
    """
    Tests error when a customer has no purchase history in cluster.
    """
    df, final_df = mock_data
    recommender.df = df
    recommender.final_df = final_df.copy()
    recommender.final_df.loc["CUSTX"] = 0  # Add new customer to cluster

    with pytest.raises(HTTPException) as exc_info:
        recommender.recommend_products("CUSTX")
    assert exc_info.value.status_code == 404
