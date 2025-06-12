import pandas as pd
import joblib
import os
import logging
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import HTTPException  # Required for raising 4xx/5xx errors

from app.config import DF_PATH, MODEL_PATH

df, final_df = None, None

def load_pickled_data():
    global df, final_df
    try:
        if os.path.exists(DF_PATH) and os.path.exists(MODEL_PATH):
            df = joblib.load(DF_PATH)
            final_df = joblib.load(MODEL_PATH)

            # Validate types after loading
            if not isinstance(df, pd.DataFrame):
                logging.error("Loaded object from DF_PATH is not a DataFrame.")
                raise ValueError("Invalid format: df must be a pandas DataFrame.")

            if not isinstance(final_df, pd.DataFrame):
                logging.error("Loaded object from MODEL_PATH is not a DataFrame.")
                raise ValueError("Invalid format: final_df must be a pandas DataFrame.")

            logging.info("Loaded pickled model and dataframe successfully.")
        else:
            logging.warning("Pickle files not found.")
            raise FileNotFoundError("Model or data file not found.")
    except Exception as e:
        logging.exception("Failed to load pickle files.")
        raise RuntimeError(f"Error loading pickled data: {e}")

def recommend_products(customer_id: str, top_n: int = 5):
    global df, final_df
    try:
        if df is None or final_df is None:
            logging.error("Recommendation attempted without loading model.")
            raise HTTPException(status_code=500, detail="Model not loaded. Please try again later.")

        # Debug-type check added
        if not isinstance(final_df, pd.DataFrame):
            raise HTTPException(status_code=500, detail="Model format is invalid. Expected a DataFrame.")

        if customer_id not in final_df.index:  # <-- ✅ this is a property
            logging.warning(f"Invalid CustomerID requested: {customer_id}")
            raise HTTPException(status_code=404, detail=f"Customer ID '{customer_id}' not found.")

        cluster = final_df.loc[customer_id, "Cluster"]
        cluster_df = df[df["Cluster"] == cluster]

        matrix = cluster_df.groupby(["Customer ID", "Product ID"])["Quantity"].sum().unstack(fill_value=0)

        if customer_id not in matrix.index:
            logging.warning(f"No product data for customer {customer_id}")
            raise HTTPException(status_code=404, detail=f"No product data found for customer '{customer_id}'.")

        sim = cosine_similarity(matrix)
        sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
        similar = sim_df[customer_id].sort_values(ascending=False).iloc[1:].index

        already_bought = set(matrix.loc[customer_id][matrix.loc[customer_id] > 0].index)
        candidates = [
            prod for cust in similar for prod in matrix.loc[cust][matrix.loc[cust] > 0].index
        ]
        ranked = sorted(Counter(candidates).items(), key=lambda x: x[1], reverse=True)

        recommendations = [p for p, _ in ranked if p not in already_bought][:top_n]

        if len(recommendations) < top_n:
            popular = cluster_df.groupby("Product ID")["Quantity"].sum().sort_values(ascending=False).index
            for p in popular:
                if p not in recommendations and p not in already_bought:
                    recommendations.append(p)
                if len(recommendations) == top_n:
                    break

        if not recommendations:
            logging.warning(f"No recommendations could be generated for customer '{customer_id}'")
            raise HTTPException(status_code=404, detail=f"No recommendations found for customer '{customer_id}'.")

        return recommendations

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        logging.exception("Error during recommendation generation.")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
