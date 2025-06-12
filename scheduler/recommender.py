"""
recommender.py

This module provides functions to:
- Prepare customer features for clustering
- Apply KMeans clustering
- Determine the optimal number of clusters using silhouette score

It outputs a processed DataFrame with cluster labels and logs each step for traceability.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import silhouette_score


def prepare_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares features from the input DataFrame for clustering.

    Steps:
    - Aggregates product quantities per customer
    - Extracts static customer features
    - One-hot encodes categorical attributes
    - Scales features using MinMaxScaler
    - Applies KMeans clustering
    - Maps cluster assignments back to the original raw_df

    Parameters:
    ----------
    raw_df : pd.DataFrame
        Raw transaction data including customer and product information.

    Returns:
    -------
    final_df_local : pd.DataFrame
        A DataFrame containing features and assigned cluster labels.
    """
    try:
        logging.info("Preparing features for clustering.")

        if raw_df.empty:
            raise ValueError("Input DataFrame is empty.")

        df = raw_df.copy()

        # Step 1: Aggregate product quantities purchased by each customer
        try:
            purchase_summary = df.groupby(["Customer ID", "Product ID"])["Quantity"].sum().unstack(fill_value=0)
        except KeyError as e:
            logging.error(f"Missing expected columns during purchase summary: {e}")
            raise

        # Step 2: Extract demographic info per customer
        try:
            customer_info = df.groupby("Customer ID").agg({
                "Customer Age": "first",
                "Customer Gender": "first",
                "Warehouse ID": "first"
            })
        except KeyError as e:
            logging.error(f"Missing expected customer info columns: {e}")
            raise

        # Step 3: One-hot encode gender and warehouse location
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = encoder.fit_transform(customer_info[["Customer Gender", "Warehouse ID"]])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(), index=customer_info.index)
        except Exception as e:
            logging.error(f"Error during encoding: {e}")
            raise

        # Step 4: Combine all features
        try:
            final_df_local = pd.concat(
                [purchase_summary, customer_info[["Customer Age"]], encoded_df],
                axis=1
            )
        except Exception as e:
            logging.error(f"Error concatenating features: {e}")
            raise

        # Step 5: Scale features
        try:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(final_df_local)
        except Exception as e:
            logging.error(f"Error during scaling: {e}")
            raise

        # Step 6: Determine optimal clusters and apply KMeans
        try:
            optimal_k = find_optimal_clusters(scaled)
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
            final_df_local["Cluster"] = kmeans.fit_predict(scaled)
        except Exception as e:
            logging.error(f"Clustering failed: {e}")
            raise

        # Step 7: Map clusters back to raw_df
        try:
            raw_df["Cluster"] = raw_df["Customer ID"].map(final_df_local["Cluster"])
        except Exception as e:
            logging.error(f"Error mapping clusters back to raw data: {e}")
            raise

        logging.info("Feature preparation and clustering completed.")
        return final_df_local

    except Exception as e:
        logging.error(f"prepare_features failed: {e}")
        raise


def find_optimal_clusters(scaled_data: np.ndarray, k_range: tuple = (2, 10)) -> int:
    """
    Finds the optimal number of clusters using silhouette score.

    Parameters:
    ----------
    scaled_data : np.ndarray
        Normalized feature matrix.
    k_range : tuple, optional
        Range of cluster counts to test (default is (2, 10)).

    Returns:
    -------
    best_k : int
        The number of clusters with the highest silhouette score.
    """
    try:
        if len(scaled_data) < 3:
            raise ValueError("Not enough data points for clustering.")

        max_k = min(k_range[1], len(scaled_data) - 1)
        best_k, best_score = 2, -1

        for k in range(k_range[0], max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                labels = kmeans.fit_predict(scaled_data)
                score = silhouette_score(scaled_data, labels)
                if score > best_score:
                    best_k, best_score = k, score
            except Exception as e:
                logging.warning(f"Silhouette calculation failed for k={k}: {e}")
                continue

        if best_score == -1:
            raise ValueError("Could not determine optimal clusters.")

        return best_k

    except Exception as e:
        logging.error(f"find_optimal_clusters failed: {e}")
        raise
