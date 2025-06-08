from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from collections import Counter
import pandas as pd
import logging

df, final_df = None, None

def prepare_features(raw_df):
    global df, final_df
    logging.info("Starting feature preparation.")
    df = raw_df.copy()

    #Customer Purchase Matrix
    purchase_summary = df.groupby(["Customer ID", "Product ID"])["Quantity"] \
                         .sum().unstack(fill_value=0)

    #Customer Demographics
    customer_info = df.groupby("Customer ID").agg({
        "Customer Age": "first",
        "Customer Gender": "first",
        "Warehouse ID": "first"
    })

    #One-Hot Encoding + Feature Join
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(customer_info[["Customer Gender", "Warehouse ID"]])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(),
        index=customer_info.index
    )

    final_df_local = pd.concat([purchase_summary, customer_info[["Customer Age"]], encoded_df], axis=1)

    #Normalization + Clustering
    scaled = MinMaxScaler().fit_transform(final_df_local)

    optimal_k = find_optimal_clusters(scaled)
    logging.info(f"Optimal clusters found: {optimal_k}")

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    final_df_local["Cluster"] = kmeans.fit_predict(scaled)
    logging.info(f"Assigned clusters for {len(final_df_local)} customers.")

    df["Cluster"] = df["Customer ID"].map(final_df_local["Cluster"])
    final_df = final_df_local
    return final_df_local

def find_optimal_clusters(scaled_data, k_range=(2, 10)):
    n_samples = len(scaled_data)
    max_k = min(k_range[1], n_samples - 1)  # Ensure max clusters < n_samples
    
    if max_k < 2:
        return 2  # Default fallback
    
    best_k = 2
    best_score = -1
    for k in range(k_range[0], max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(scaled_data)
        score = silhouette_score(scaled_data, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def recommend_products(customer_id: str, top_n=5):
    global df, final_df
    logging.info(f"Generating recommendations for Customer ID: {customer_id}")

    #Check if the customer exists
    if customer_id not in final_df.index:
        logging.warning(f"Customer ID {customer_id} not found in data.")
        return ["Customer ID not found"]

    # Find Cluster Neighbors
    cluster = final_df.loc[customer_id, "Cluster"]
    cluster_df = df[df["Cluster"] == cluster]

    matrix = cluster_df.groupby(["Customer ID", "Product ID"])["Quantity"] \
                       .sum().unstack(fill_value=0)

    if customer_id not in matrix.index:
        logging.warning(f"No purchase data for customer {customer_id}.")
        return ["No product data for this customer"]

    #Cosine Similarity
    sim = cosine_similarity(matrix)
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)

    similar = sim_df[customer_id].sort_values(ascending=False).iloc[1:100].index

    # Get Products the Customer Already Bought
    already_bought = set(matrix.loc[customer_id][matrix.loc[customer_id] > 0].index)
    #Get Candidate Products from Similar Customers
    candidates = [prod for cust in similar for prod in matrix.loc[cust][matrix.loc[cust] > 0].index]

    #Rank Candidate Products by Popularity (among similar users)
    ranked = sorted(Counter(candidates).items(), key=lambda x: x[1], reverse=True)

    #Select Top N Products Not Already Bought
    recommendations = [p for p, _ in ranked if p not in already_bought][:top_n]

    #Fill Remaining Slots with Popular Items Within The Cluster
    if len(recommendations) < top_n:
        popular = cluster_df.groupby("Product ID")["Quantity"] \
                            .sum().sort_values(ascending=False).index
        for p in popular:
            if p not in recommendations and p not in already_bought:
                recommendations.append(p)
            if len(recommendations) == top_n:
                break

    logging.info(f"Recommendations for {customer_id}: {recommendations}")
    return recommendations
