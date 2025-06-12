"""
test_recommender.py

Unit tests for scheduler.recommender module functions including:
- prepare_features
- find_optimal_clusters

These tests ensure correct clustering logic, error handling, and data transformation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from scheduler import recommender


@pytest.fixture
def mock_raw_df():
    """
    Fixture to create a synthetic raw dataframe for clustering tests.
    """
    return pd.DataFrame({
        "Customer ID": ["C1", "C1", "C2", "C3", "C3"],
        "Product ID": ["P1", "P2", "P1", "P2", "P3"],
        "Quantity": [1, 2, 3, 4, 1],
        "Customer Age": [30, 30, 45, 25, 25],
        "Customer Gender": ["M", "M", "F", "F", "F"],
        "Warehouse ID": ["W1", "W1", "W2", "W3", "W3"]
    })


def test_prepare_features_success(mock_raw_df):
    """
    Test prepare_features returns expected structure with valid cluster labels.
    """
    output_df = recommender.prepare_features(mock_raw_df.copy())

    # Assertions
    assert "Cluster" in output_df.columns
    assert isinstance(output_df, pd.DataFrame)
    assert len(output_df) == len(mock_raw_df["Customer ID"].unique())
    assert output_df["Cluster"].dtype in [np.int32, np.int64]


def test_prepare_features_empty_input():
    """
    Test prepare_features raises ValueError on empty input.
    """
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        recommender.prepare_features(empty_df)


@patch("scheduler.recommender.silhouette_score", return_value=0.8)
@patch("scheduler.recommender.KMeans")
def test_find_optimal_clusters_valid(mock_kmeans, mock_silhouette):
    """
    Test find_optimal_clusters returns a valid K within range for proper input.
    """
    # Fake scaled data
    X = np.random.rand(20, 5)

    best_k = recommender.find_optimal_clusters(X, k_range=(2, 5))
    assert isinstance(best_k, int)
    assert 2 <= best_k <= 5


def test_find_optimal_clusters_insufficient_data():
    """
    Test find_optimal_clusters raises error if too few data points.
    """
    X_small = np.array([[0.1, 0.2]])
    with pytest.raises(ValueError, match="Not enough data points for clustering"):
        recommender.find_optimal_clusters(X_small)


@patch("scheduler.recommender.silhouette_score", side_effect=Exception("Silhouette failure"))
@patch("scheduler.recommender.KMeans")
def test_find_optimal_clusters_all_failures(mock_kmeans, mock_silhouette):
    """
    Test fallback behavior if all silhouette computations fail.
    """
    X = np.random.rand(10, 3)
    with pytest.raises(ValueError, match="Could not determine optimal clusters"):
        recommender.find_optimal_clusters(X, k_range=(2, 3))
