import pandas as pd
import joblib
import os
import logging
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from app.config import DF_PATH, MODEL_PATH

df, final_df = None, None

def load_pickled_data():
    global df, final_df
    if os.path.exists(DF_PATH) and os.path.exists(MODEL_PATH):
        df = joblib.load(DF_PATH)
        final_df = joblib.load(MODEL_PATH)
        logging.info("Loaded pickled model.")
    else:
        logging.warning("Pickle files not found.")

def recommend_products(customer_id: str, top_n: int = 5):
    global df, final_df
    if df is None or final_df is None:
        return ["Model not loaded"]

    if customer_id not in final_df.index:
        return ["Customer ID not found"]

    cluster = final_df.loc[customer_id, "Cluster"]
    cluster_df = df[df["Cluster"] == cluster]

    matrix = cluster_df.groupby(["Customer ID", "Product ID"])["Quantity"].sum().unstack(fill_value=0)

    if customer_id not in matrix.index:
        return ["No product data for this customer"]

    sim = cosine_similarity(matrix)
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
    similar = sim_df[customer_id].sort_values(ascending=False).iloc[1:].index

    already_bought = set(matrix.loc[customer_id][matrix.loc[customer_id] > 0].index)
    candidates = [prod for cust in similar for prod in matrix.loc[cust][matrix.loc[cust] > 0].index]
    ranked = sorted(Counter(candidates).items(), key=lambda x: x[1], reverse=True)

    recommendations = [p for p, _ in ranked if p not in already_bought][:top_n]

    if len(recommendations) < top_n:
        popular = cluster_df.groupby("Product ID")["Quantity"].sum().sort_values(ascending=False).index
        for p in popular:
            if p not in recommendations and p not in already_bought:
                recommendations.append(p)
            if len(recommendations) == top_n:
                break

    return recommendations
