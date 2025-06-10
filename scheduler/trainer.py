import pandas as pd
import time
import logging
import os
import pickle
from datetime import datetime, timedelta
from app.config import CSV_PATH, RETRAIN_INTERVAL, ROW_GROWTH_THRESHOLD, MODEL_DIR
from app.recommender import prepare_features

# Global state
df = pd.DataFrame()
final_df = pd.DataFrame()
last_retrain_time = time.time()
last_row_count = 0

def save_model_with_timestamp(df, final_df):
    """
    Save df and final_df with timestamp, and as 'latest' versions.
    Also cleans up old timestamped files.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # File paths
    df_ts_path = os.path.join(MODEL_DIR, f"df_{timestamp}.pkl")
    final_df_ts_path = os.path.join(MODEL_DIR, f"final_df_{timestamp}.pkl")
    df_latest_path = os.path.join(MODEL_DIR, "df.pkl")
    final_df_latest_path = os.path.join(MODEL_DIR, "final_df.pkl")

    # Save timestamped
    with open(df_ts_path, "wb") as f:
        pickle.dump(df, f)
    with open(final_df_ts_path, "wb") as f:
        pickle.dump(final_df, f)

    # Save latest (overwrite)
    with open(df_latest_path, "wb") as f:
        pickle.dump(df, f)
    with open(final_df_latest_path, "wb") as f:
        pickle.dump(final_df, f)

    logging.info(f"Models saved: {df_ts_path}, {final_df_ts_path}")
    cleanup_old_pickles(MODEL_DIR)

def cleanup_old_pickles(directory, retention_days=365):
    """
    Delete timestamped pickle files older than `retention_days`.
    Keeps only recent models.
    """
    cutoff = datetime.now() - timedelta(days=retention_days)

    for filename in os.listdir(directory):
        if filename.startswith(("df_", "final_df_")) and filename.endswith(".pkl"):
            try:
                timestamp_str = filename.split("_")[1].replace(".pkl", "")
                file_time = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                if file_time < cutoff:
                    os.remove(os.path.join(directory, filename))
                    logging.info(f"Deleted old model: {filename}")
            except Exception as e:
                logging.warning(f"Skipping {filename}: {e}")

def retrain_model():
    """
    Retrain the model if sufficient time or data change has occurred.
    """
    global df, final_df, last_retrain_time, last_row_count

    # Load latest CSV
    df = pd.read_csv(CSV_PATH)

    # Prepare features
    final_df = prepare_features(df)

    # Save model
    save_model_with_timestamp(df, final_df)

    # Update retrain state
    last_retrain_time = time.time()
    last_row_count = len(df)

    logging.info("Model retrained and saved.")

def maybe_retrain_model():
    """
    Check whether retraining is needed based on time or row count.
    """
    global last_row_count

    current_time = time.time()
    new_data = pd.read_csv(CSV_PATH)

    if (current_time - last_retrain_time > RETRAIN_INTERVAL or
        len(new_data) - last_row_count > ROW_GROWTH_THRESHOLD):
        retrain_model()
