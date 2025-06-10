import pandas as pd
import numpy as np
import pickle
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import logging
from app.config import MODEL_DIR

df = None
final_df = None

def load_pickled_data():
    global df, final_df

    df_path = os.path.join(MODEL_DIR, "df.pkl")
    final_df_path = os.path.join(MODEL_DIR, "final_df.pkl")

    if not os.path.exists(df_path) or not os.path.exists(final_df_path):
        print("Pickle files not found. Run retraining first.")
        return

    with open(df_path, "rb") as f:
        df = pickle.load(f)

    with open(final_df_path, "rb") as f:
        final_df = pickle.load(f)

    print("Pickled model data loaded.")

def prepare_features(raw_df):
    """
    Prepare features for clustering from raw transaction data.
    """
    global df, final_df
    logging.info("Starting feature preparation.")
    
    df = raw_df.copy()

    purchase_summary = df.groupby(["Customer ID", "Product ID"])["Quantity"] \
                         .sum().unstack(fill_value=0)

    customer_info = df.groupby("Customer ID").agg({
        "Customer Age": "first",
        "Customer Gender": "first",
        "Warehouse ID": "first"
    })

    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(customer_info[["Customer Gender", "Warehouse ID"]])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(),
        index=customer_info.index
    )

    final_df_local = pd.concat([purchase_summary, customer_info[["Customer Age"]], encoded_df], axis=1)

    scaled = MinMaxScaler().fit_transform(final_df_local)

    optimal_k = find_optimal_clusters(scaled)
    logging.info(f"Optimal clusters found: {optimal_k}")

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    final_df_local["Cluster"] = kmeans.fit_predict(scaled)

    df["Cluster"] = df["Customer ID"].map(final_df_local["Cluster"])

    final_df = final_df_local

    return final_df_local

def find_optimal_clusters(scaled_data, k_range=(2, 10)):
    n_samples = len(scaled_data)
    max_k = min(k_range[1], n_samples - 1)  
    if max_k < 2:
        return 2  
    
    best_k = 2
    best_score = -1
    
    for k in range(k_range[0], max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(scaled_data)
        score = silhouette_score(scaled_data, labels)
        if score > best_score:
            best_k, best_score = k, score
            
    return best_k
