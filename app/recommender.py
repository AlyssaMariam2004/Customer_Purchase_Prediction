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

df, final_df = None, None  # Global variables to hold dataframes for clustering and recommendations

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

def recommend_products(customer_id: str, top_n=5):
    """
    Generate product recommendations for a given customer based on cluster similarity.

    Steps:
    - Verify customer exists in clustered data.
    - Identify the customer's cluster and customers in that cluster.
    - Compute cosine similarity of purchase patterns within cluster.
    - Find top similar customers and products they bought.
    - Exclude products already bought by the customer.
    - Rank candidate products by popularity among similar customers.
    - If needed, fill remaining recommendations with popular cluster products.

    Args:
        customer_id (str): Customer identifier to generate recommendations for.
        top_n (int): Number of top recommended products to return (default 5).

    Returns:
        list: List of product IDs recommended for the customer.
              Returns error messages as list if customer not found or no data.
    """
    global df, final_df
    logging.info(f"Generating recommendations for Customer ID: {customer_id}")

    # Check if customer exists in aggregated cluster data
    if customer_id not in final_df.index:
        logging.warning(f"Customer ID {customer_id} not found in data.")
        return ["Customer ID not found"]

    # Get cluster label of the target customer
    cluster = final_df.loc[customer_id, "Cluster"]
    
    # Filter all transactions belonging to the same cluster
    cluster_df = df[df["Cluster"] == cluster]

    # Create purchase matrix for cluster customers: rows=customers, columns=products, values=quantity
    matrix = cluster_df.groupby(["Customer ID", "Product ID"])["Quantity"] \
                       .sum().unstack(fill_value=0)

    # Confirm that the customer has purchase data in the matrix
    if customer_id not in matrix.index:
        logging.warning(f"No purchase data for customer {customer_id}.")
        return ["No product data for this customer"]

    # Compute cosine similarity between customers based on purchase patterns
    sim = cosine_similarity(matrix)
    
    # Create DataFrame from similarity matrix for easier querying
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)

    # Get most similar customers excluding the customer itself (first entry)
    similar = sim_df[customer_id].sort_values(ascending=False).iloc[1:100].index

    # Set of products already bought by the customer to exclude from recommendations
    already_bought = set(matrix.loc[customer_id][matrix.loc[customer_id] > 0].index)
    
    # Gather all products bought by similar customers as candidates
    candidates = [prod for cust in similar for prod in matrix.loc[cust][matrix.loc[cust] > 0].index]

    # Rank candidate products by frequency of purchase among similar customers
    ranked = sorted(Counter(candidates).items(), key=lambda x: x[1], reverse=True)

    # Select top N products that the customer hasn't bought yet
    recommendations = [p for p, _ in ranked if p not in already_bought][:top_n]

    # If fewer than top_n recommendations, fill remaining slots with popular cluster products
    if len(recommendations) < top_n:
        # Popular products in the cluster by total quantity sold
        popular = cluster_df.groupby("Product ID")["Quantity"] \
                            .sum().sort_values(ascending=False).index
        for p in popular:
            # Add product if not already recommended or purchased
            if p not in recommendations and p not in already_bought:
                recommendations.append(p)
            if len(recommendations) == top_n:
                break

    logging.info(f"Recommendations for {customer_id}: {recommendations}")
    
    return recommendations
