import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
import logging

def prepare_features(raw_df):
    logging.info("Preparing features for clustering.")
    df = raw_df.copy()

    purchase_summary = df.groupby(["Customer ID", "Product ID"])["Quantity"].sum().unstack(fill_value=0)
    customer_info = df.groupby("Customer ID").agg({
        "Customer Age": "first",
        "Customer Gender": "first",
        "Warehouse ID": "first"
    })

    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(customer_info[["Customer Gender", "Warehouse ID"]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(), index=customer_info.index)

    final_df_local = pd.concat([purchase_summary, customer_info[["Customer Age"]], encoded_df], axis=1)
    scaled = MinMaxScaler().fit_transform(final_df_local)

    k = find_optimal_clusters(scaled)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    final_df_local["Cluster"] = kmeans.fit_predict(scaled)

    raw_df["Cluster"] = raw_df["Customer ID"].map(final_df_local["Cluster"])
    return raw_df, final_df_local

def find_optimal_clusters(scaled_data, k_range=(2, 10)):
    max_k = min(k_range[1], len(scaled_data) - 1)
    best_k, best_score = 2, -1

    for k in range(k_range[0], max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(scaled_data)
        score = silhouette_score(scaled_data, labels)
        if score > best_score:
            best_k, best_score = k, score

    return best_k

