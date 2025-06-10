import pandas as pd
import time
import logging
import os
import joblib
from datetime import datetime, timedelta
from app.config import CSV_PATH, RETRAIN_INTERVAL, ROW_GROWTH_THRESHOLD, MODEL_DIR, DF_PATH, MODEL_PATH
from app.recommender import prepare_features, load_pickled_data

last_retrain_time = time.time()
last_row_count = 0

def save_model_with_timestamp(df, final_df):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df_ts_path = os.path.join(MODEL_DIR, f"df_{timestamp}.pkl")
    final_df_ts_path = os.path.join(MODEL_DIR, f"final_df_{timestamp}.pkl")

    # Save timestamped
    joblib.dump(df, df_ts_path)
    joblib.dump(final_df, final_df_ts_path)

    # Save latest
    joblib.dump(df, DF_PATH)
    joblib.dump(final_df, MODEL_PATH)

    logging.info(f"Models saved: {df_ts_path}, {final_df_ts_path}")
    cleanup_old_pickles(MODEL_DIR)

def cleanup_old_pickles(directory, retention_days=365):
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
    global last_retrain_time, last_row_count

    df_raw = pd.read_csv(CSV_PATH)
    final_df_prepared = prepare_features(df_raw)

    save_model_with_timestamp(df_raw, final_df_prepared)

    # Load into memory for use after retraining
    from app.recommender import load_pickled_data
    load_pickled_data()

    last_retrain_time = time.time()
    last_row_count = len(df_raw)

    logging.info("Model retrained, saved, and loaded.")

def maybe_retrain_model():
    global last_row_count

    current_time = time.time()
    new_data = pd.read_csv(CSV_PATH)

    if (current_time - last_retrain_time > RETRAIN_INTERVAL or
        len(new_data) - last_row_count > ROW_GROWTH_THRESHOLD):
        retrain_model()
        load_pickled_data() 