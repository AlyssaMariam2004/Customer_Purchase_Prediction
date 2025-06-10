from collections import Counter
import pandas as pd
import logging
import pickle
from app.config import MODEL_DIR
from sklearn.metrics.pairwise import cosine_similarity

with open(f"{MODEL_DIR}/df.pkl", "rb") as f:
    df = pickle.load(f)

with open(f"{MODEL_DIR}/final_df.pkl", "rb") as f:
    final_df = pickle.load(f)

# Use df, final_df for recommendations only


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
