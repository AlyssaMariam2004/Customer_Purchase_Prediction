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

def save_model(df, final_df):
    try:
        joblib.dump(df, DF_PATH)
        joblib.dump(final_df, MODEL_PATH)
        logging.info("Model saved to DF_PATH and MODEL_PATH.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        joblib.dump(df, os.path.join(MODEL_DIR, f"df_{timestamp}.pkl"))
        joblib.dump(final_df, os.path.join(MODEL_DIR, f"final_df_{timestamp}.pkl"))
        cleanup_old_pickles()
    except Exception as e:
        logging.error(f"Error while saving models: {e}")

def cleanup_old_pickles(keep=1):
    try:
        def valid(p): return p.endswith(".pkl") and ("df_" in p or "final_df_" in p)
        all_files = sorted([f for f in os.listdir(MODEL_DIR) if valid(f)], reverse=True)
        to_remove = all_files[keep * 2:]
        for f in to_remove:
            path = os.path.join(MODEL_DIR, f)
            os.remove(path)
            logging.info(f"Removed old model backup: {f}")
    except Exception as e:
        logging.error(f"Error during model cleanup: {e}")

def retrain_model():
    global last_retrain_time, last_row_count

    try:
        if not os.path.exists(CSV_PATH):
            logging.warning("CSV file not found. Cannot retrain model.")
            return

        df_raw = pd.read_csv(CSV_PATH)
        if df_raw.empty:
            logging.warning("CSV file is empty. Cannot retrain model.")
            return

        final_df = prepare_features(df_raw)
        save_model(df_raw, final_df)

        last_retrain_time = time.time()
        last_row_count = len(df_raw)
        logging.info("Model retrained successfully.")
    except Exception as e:
        logging.error(f"Error during retrain_model: {e}")

def maybe_retrain_model():
    global last_row_count
    try:
        if not os.path.exists(CSV_PATH):
            logging.warning("CSV file not found. Cannot check for retraining.")
            return

        df = pd.read_csv(CSV_PATH)
        if df.empty:
            logging.warning("CSV file is empty. Skipping retraining check.")
            return

        if time.time() - last_retrain_time > RETRAIN_INTERVAL or len(df) - last_row_count > ROW_GROWTH_THRESHOLD:
            retrain_model()
        else:
            logging.info("No retraining needed at this time.")
    except Exception as e:
        logging.error(f"Error in maybe_retrain_model: {e}")
