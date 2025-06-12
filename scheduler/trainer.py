import os
import time
import logging
import joblib
import pandas as pd
from datetime import datetime
from config import CSV_PATH, RETRAIN_INTERVAL, ROW_GROWTH_THRESHOLD, MODEL_DIR, DF_PATH, MODEL_PATH
from recommender import prepare_features

last_retrain_time = time.time()
last_row_count = 0

def save_model(df_raw, final_df):
    joblib.dump(df_raw, DF_PATH)
    joblib.dump(final_df, MODEL_PATH)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(df_raw, os.path.join(MODEL_DIR, f"df_{timestamp}.pkl"))
    joblib.dump(final_df, os.path.join(MODEL_DIR, f"final_df_{timestamp}.pkl"))

    cleanup_old_pickles()

def cleanup_old_pickles(keep=1):
    def is_model_file(f): return f.endswith(".pkl") and ("df_" in f or "final_df_" in f)
    files = sorted([f for f in os.listdir(MODEL_DIR) if is_model_file(f)], reverse=True)
    to_remove = files[keep*2:]
    for f in to_remove:
        os.remove(os.path.join(MODEL_DIR, f))

def retrain_model():
    global last_retrain_time, last_row_count
    df_raw = pd.read_csv(CSV_PATH)
    df_with_cluster, final_df = prepare_features(df_raw)
    save_model(df_with_cluster, final_df)
    last_retrain_time = time.time()
    last_row_count = len(df_raw)
    logging.info("Model retrained.")

def maybe_retrain_model():
    df = pd.read_csv(CSV_PATH)
    if time.time() - last_retrain_time > RETRAIN_INTERVAL or len(df) - last_row_count > ROW_GROWTH_THRESHOLD:
        retrain_model()
