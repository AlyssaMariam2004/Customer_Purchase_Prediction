"""
recommender.py

This module handles:
- Loading pickled model and data files used for recommendation.
- Generating product recommendations based on customer purchase similarity using cosine similarity within clusters.
"""

import os
import joblib
import logging
import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import HTTPException

from app.config import DATAFRAME_PATH, MODEL_FILE_PATH

# Global variables for the loaded datasets
df = None  # Complete purchase data
final_df = None  # Customer-to-cluster mapping

def load_pickled_data() -> None:
    """
    Loads the pickled DataFrame and model (cluster-labeled DataFrame) into global variables.

    Raises:
        RuntimeError: If loading fails or file formats are invalid.
    """
    global df, final_df
    try:
        if os.path.exists(DATAFRAME_PATH) and os.path.exists(MODEL_FILE_PATH):
            df = joblib.load(DATAFRAME_PATH)
            final_df = joblib.load(MODEL_FILE_PATH)

            # Type validation
            if not isinstance(df, pd.DataFrame):
                logging.error("Loaded object from DATAFRAME_PATH is not a DataFrame.")
                raise ValueError("df must be a pandas DataFrame.")

            if not isinstance(final_df, pd.DataFrame):
                logging.error("Loaded object from MODEL_FILE_PATH is not a DataFrame.")
                raise ValueError("final_df must be a pandas DataFrame.")

            logging.info("Successfully loaded pickled data and model.")
        else:
            logging.warning("Pickle files not found.")
            raise FileNotFoundError("Pickled model or data file not found.")
    except Exception as e:
        logging.exception("Failed to load pickled files.")
        raise RuntimeError(f"Error loading pickled data: {e}")


def recommend_products(customer_id: str, top_n: int = 5) -> list:
    """
    Generates a list of recommended products for a given customer using cosine similarity.

    Args:
        customer_id (str): ID of the customer requesting recommendations.
        top_n (int): Number of recommendations to return.

    Returns:
        list: List of recommended Product IDs.

    Raises:
        HTTPException: If model is not loaded, customer is not found, or other errors occur.
    """
    global df, final_df

    try:
        if df is None or final_df is None:
            logging.error("Model data not loaded before recommendation request.")
            raise HTTPException(status_code=500, detail="Model not loaded. Please try again later.")

        if not isinstance(final_df, pd.DataFrame):
            raise HTTPException(status_code=500, detail="Invalid model format. Expected DataFrame.")

        # Validate customer ID presence in model
        if customer_id not in final_df.index:
            logging.warning(f"Requested Customer ID not found: {customer_id}")
            raise HTTPException(status_code=404, detail=f"Customer ID '{customer_id}' not found.")

        # Retrieve customer's cluster and filter data
        cluster = final_df.loc[customer_id, "Cluster"]
        cluster_df = df[df["Cluster"] == cluster]

        # Build customer-product matrix
        matrix = (
            cluster_df
            .groupby(["Customer ID", "Product ID"])["Quantity"]
            .sum()
            .unstack(fill_value=0)
        )

        if customer_id not in matrix.index:
            logging.warning(f"No purchase history for customer {customer_id}")
            raise HTTPException(status_code=404, detail=f"No product data for customer '{customer_id}'.")

        # Compute cosine similarity between customers in the same cluster
        similarity_matrix = cosine_similarity(matrix)
        similarity_df = pd.DataFrame(similarity_matrix, index=matrix.index, columns=matrix.index)
        similar_customers = similarity_df[customer_id].sort_values(ascending=False).iloc[1:].index

        # Products the customer already bought
        already_bought = set(matrix.loc[customer_id][matrix.loc[customer_id] > 0].index)

        # Collect candidate products from similar customers
        candidate_products = [
            product
            for other_customer in similar_customers
            for product in matrix.loc[other_customer][matrix.loc[other_customer] > 0].index
        ]

        # Rank candidate products by frequency
        ranked_candidates = sorted(Counter(candidate_products).items(), key=lambda x: x[1], reverse=True)

        # Filter out already bought products
        recommendations = [prod for prod, _ in ranked_candidates if prod not in already_bought][:top_n]

        # Fallback to popular products if not enough recommendations
        if len(recommendations) < top_n:
            popular_products = (
                cluster_df.groupby("Product ID")["Quantity"]
                .sum()
                .sort_values(ascending=False)
                .index
            )
            for product_id in popular_products:
                if product_id not in recommendations and product_id not in already_bought:
                    recommendations.append(product_id)
                if len(recommendations) == top_n:
                    break

        # Final check
        if not recommendations:
            logging.warning(f"No recommendations could be generated for customer '{customer_id}'")
            raise HTTPException(status_code=404, detail=f"No recommendations found for customer '{customer_id}'.")

        return recommendations

    except HTTPException as http_err:
        raise http_err  # Forward known HTTP errors
    except Exception as e:
        logging.exception("Unexpected error during recommendation generation.")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

