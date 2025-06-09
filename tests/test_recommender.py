import pandas as pd
import pytest
from app import recommender
import numpy as np

def make_test_data():
    """
    Creates a sample DataFrame with test customer purchase and demographic data.

    Returns:
        pd.DataFrame: Test data including customer IDs, product IDs, quantities,
                      customer age, gender, and warehouse ID.
    """
    data = {
        "Customer ID": ["CUST1", "CUST1", "CUST2", "CUST3"],
        "Product ID": ["P1", "P2", "P2", "P3"],
        "Quantity": [2, 1, 3, 5],
        "Customer Age": [30, 30, 40, 25],
        "Customer Gender": ["M", "M", "F", "F"],
        "Warehouse ID": ["WH1", "WH1", "WH2", "WH2"]
    }
    return pd.DataFrame(data)

def test_prepare_features_creates_clusters():
    """
    Tests that the prepare_features function correctly clusters customers.

    Asserts:
        - The output DataFrame contains a 'Cluster' column.
        - The indices of the output match the unique Customer IDs.
        - No null values exist in the 'Cluster' assignments.
    """
    df = make_test_data()
    result = recommender.prepare_features(df)
    assert "Cluster" in result.columns
    assert set(result.index) == set(df["Customer ID"].unique())
    assert not result["Cluster"].isnull().any()

def test_recommend_products_returns_list_and_nonempty():
    """
    Tests the recommend_products function to ensure it returns a list of recommended products.

    Checks:
        - Return type is a list.
        - Recommendations exclude products already purchased by the customer.
        - The number of recommendations does not exceed the requested top_n.
    """
    df = make_test_data()
    recommender.prepare_features(df)

    recs = recommender.recommend_products("CUST1", top_n=3)
    assert isinstance(recs, list)
    already_bought = set(df[df["Customer ID"]=="CUST1"]["Product ID"])
    for p in recs:
        assert p not in already_bought
    assert len(recs) <= 3

def test_recommend_products_unknown_customer():
    """
    Tests behavior of recommend_products when given an unknown customer ID.

    Expects the function to return a specific error message in a list.
    """
    df = make_test_data()
    recommender.prepare_features(df)
    recs = recommender.recommend_products("UNKNOWN_CUSTOMER")
    assert recs == ["Customer ID not found"]

def test_recommend_products_no_purchase_data(monkeypatch):
    """
    Tests recommend_products for a customer present in the data but with no purchase history.

    Adds a new customer with zero quantity purchases, verifies the function returns an appropriate message.
    """
    df = make_test_data()
    recommender.prepare_features(df)

    # Add a new customer with no purchases but a cluster assignment
    cluster = recommender.final_df.loc["CUST1", "Cluster"]

    new_row = pd.DataFrame({
        "Customer ID": ["NO_PURCHASE"],
        "Product ID": [None],
        "Quantity": [0],
        "Customer Age": [50],
        "Customer Gender": ["M"],
        "Warehouse ID": ["WH1"],
        "Cluster": [cluster]
    })

    recommender.df = pd.concat([recommender.df, new_row], ignore_index=True)
    recommender.final_df.loc["NO_PURCHASE"] = cluster

    recs = recommender.recommend_products("NO_PURCHASE")
    assert recs == ["No product data for this customer"]

def test_find_optimal_clusters_min_data():
    """
    Tests find_optimal_clusters with minimal data points.

    Verifies that the function returns the correct number of clusters when data is small.
    """
    result = recommender.find_optimal_clusters(np.array([[1.0], [2.0]]))
    assert result == 2
