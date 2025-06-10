"""
Customer Purchase Clustering and Recommendation Module.

This module prepares customer purchase and demographic features,
finds optimal clusters of customers using KMeans and silhouette score,
and generates product recommendations based on cluster similarity.

Global variables:
- df: Raw transaction dataframe with cluster assignments.
- final_df: Dataframe containing aggregated customer features and clusters.
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from collections import Counter
import pandas as pd
import logging
import joblib
from app.config import MODEL_PATH

# Load model
try:
    final_df = joblib.load(MODEL_PATH)
except Exception as e:
    final_df = None
    print(f"Error loading model: {e}")


def prepare_features(raw_df):
    """
    Prepare features for clustering from raw transaction data.

    Steps:
    - Summarize purchase quantities per customer-product.
    - Aggregate customer demographics.
    - One-hot encode categorical demographic features.
    - Normalize all features.
    - Determine optimal number of clusters.
    - Assign cluster labels to customers.

    Args:
        raw_df (pd.DataFrame): Raw transaction data containing customer, product,
                               quantity, demographics, etc.

    Returns:
        pd.DataFrame: Aggregated customer features with cluster assignments.
    """
    global df, final_df
    logging.info("Starting feature preparation.")
    
    # Make a copy of the input data to avoid modifying original
    df = raw_df.copy()

    # Create a purchase matrix: rows=customers, columns=products, values=total quantity bought
    purchase_summary = df.groupby(["Customer ID", "Product ID"])["Quantity"] \
                         .sum().unstack(fill_value=0)

    # Aggregate demographic info per customer (taking first occurrence as representative)
    customer_info = df.groupby("Customer ID").agg({
        "Customer Age": "first",
        "Customer Gender": "first",
        "Warehouse ID": "first"
    })

    # One-Hot encode categorical features "Customer Gender" and "Warehouse ID"
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(customer_info[["Customer Gender", "Warehouse ID"]])
    
    # Convert encoded numpy array back to a DataFrame with proper column names and index
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(),
        index=customer_info.index
    )

    # Combine purchase data, numeric demographics, and encoded categorical features into one DataFrame
    final_df_local = pd.concat([purchase_summary, customer_info[["Customer Age"]], encoded_df], axis=1)

    # Normalize all feature values to the 0-1 range for clustering stability
    scaled = MinMaxScaler().fit_transform(final_df_local)

    # Determine optimal number of clusters using silhouette score within a range of cluster counts
    optimal_k = find_optimal_clusters(scaled)
    logging.info(f"Optimal clusters found: {optimal_k}")

    # Create and fit KMeans clustering model with the optimal cluster count
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    
    # Assign cluster labels to each customer and add as a new column
    final_df_local["Cluster"] = kmeans.fit_predict(scaled)
    logging.info(f"Assigned clusters for {len(final_df_local)} customers.")

    # Map cluster labels back to original transaction data by Customer ID for downstream usage
    df["Cluster"] = df["Customer ID"].map(final_df_local["Cluster"])
    
    # Store final aggregated DataFrame with clusters as a global variable
    final_df = final_df_local
    
    return final_df_local

def find_optimal_clusters(scaled_data, k_range=(2, 10)):
    """
    Find the optimal number of clusters (k) for KMeans using silhouette score.

    Args:
        scaled_data (np.ndarray): Normalized feature matrix for clustering.
        k_range (tuple): Tuple of (min_k, max_k) to search for optimal clusters.

    Returns:
        int: Optimal number of clusters between k_range[0] and k_range[1].
    """
    n_samples = len(scaled_data)
    
    # Limit max clusters to fewer than total samples to avoid errors
    max_k = min(k_range[1], n_samples - 1)  
    
    # If not enough samples, fallback to 2 clusters
    if max_k < 2:
        return 2  
    
    best_k = 2
    best_score = -1
    
    # Try each k in the range and compute silhouette score
    for k in range(k_range[0], max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(scaled_data)
        
        # Silhouette score measures how well samples fit their assigned cluster
        score = silhouette_score(scaled_data, labels)
        
        # Keep track of best cluster count based on highest silhouette score
        if score > best_score:
            best_k, best_score = k, score
            
    return best_k