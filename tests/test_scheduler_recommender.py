import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scheduler import recommender


@pytest.fixture
def sample_raw_df():
    """
    Provides a valid mock DataFrame with at least 3 customers,
    ensuring silhouette score calculation can be performed.
    """
    return pd.DataFrame({
        "Customer ID": ["CUST1", "CUST1", "CUST2", "CUST3"],
        "Product ID": ["Product_101", "Product_102", "Product_101", "Product_103"],
        "Quantity": [2, 1, 5, 3],
        "Customer Age": [30, 30, 40, 35],
        "Customer Gender": ["M", "M", "F", "M"],
        "Warehouse ID": ["W1", "W1", "W2", "W3"]
    })


def test_prepare_features_success(sample_raw_df):
    """
    Positive Test:
    Verifies that prepare_features returns a DataFrame with a 'Cluster' column
    when valid input is provided.
    """
    result_df = recommender.prepare_features(sample_raw_df)
    assert isinstance(result_df, pd.DataFrame)
    assert "Cluster" in result_df.columns
    assert result_df.shape[0] == 3  # 3 unique customers


def test_prepare_features_empty_input():
    """
    Negative Test:
    Verifies that prepare_features raises ValueError when given an empty DataFrame.
    """
    with pytest.raises(ValueError, match="Input DataFrame is empty."):
        recommender.prepare_features(pd.DataFrame())


def test_find_optimal_clusters_valid_data():
    """
    Positive Test:
    Verifies that a valid number of clusters is returned for proper scaled input.
    """
    data = np.array([[0.1, 0.2], [0.2, 0.3], [0.9, 0.8], [0.95, 0.85]])
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    best_k = recommender.find_optimal_clusters(scaled_data)
    assert isinstance(best_k, int)
    assert 2 <= best_k <= len(scaled_data) - 1


def test_find_optimal_clusters_insufficient_data():
    """
    Negative Test:
    Verifies that ValueError is raised if less than 3 samples are provided for clustering.
    """
    scaled = np.array([[0.1, 0.2], [0.3, 0.4]])  # Only 2 samples
    with pytest.raises(ValueError, match="Not enough data points for clustering."):
        recommender.find_optimal_clusters(scaled)



