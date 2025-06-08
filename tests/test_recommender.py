import pandas as pd
import pytest
from app import recommender
import numpy as np

def make_test_data():
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
    df = make_test_data()
    result = recommender.prepare_features(df)
    # Clusters column should exist and have same index as customers
    assert "Cluster" in result.columns
    assert set(result.index) == set(df["Customer ID"].unique())
    assert not result["Cluster"].isnull().any()

def test_recommend_products_returns_list_and_nonempty():
    df = make_test_data()
    recommender.prepare_features(df)

    # Test for a known customer
    recs = recommender.recommend_products("CUST1", top_n=3)
    assert isinstance(recs, list)
    # Recommendations should not include products already bought by CUST1 (P1, P2)
    already_bought = set(df[df["Customer ID"]=="CUST1"]["Product ID"])
    for p in recs:
        assert p not in already_bought
    # Length <= top_n
    assert len(recs) <= 3

def test_recommend_products_unknown_customer():
    df = make_test_data()
    recommender.prepare_features(df)

    recs = recommender.recommend_products("UNKNOWN_CUSTOMER")
    assert recs == ["Customer ID not found"]

def test_recommend_products_no_purchase_data(monkeypatch):
    df = make_test_data()
    recommender.prepare_features(df)

    # Add a customer with no purchase data in global df
    recommender.df = pd.concat([recommender.df, pd.DataFrame({
        "Customer ID": ["NO_PURCHASE"],
        "Product ID": [None],
        "Quantity": [0],
        "Customer Age": [50],
        "Customer Gender": ["M"],
        "Warehouse ID": ["WH1"],
        "Cluster": [recommender.final_df.loc["CUST1", "Cluster"]]
    })], ignore_index=True)

    recs = recommender.recommend_products("NO_PURCHASE")
    assert recs == ["No product data for this customer"]

def test_recommend_products_no_purchase_data():
    result = recommender.recommend_products("NON_EXISTENT_CUSTOMER")
    assert result == ["Customer ID not found"]

def test_find_optimal_clusters_min_data():
    result = recommender.find_optimal_clusters(np.array([[1.0], [2.0]]))
    assert result == 2
