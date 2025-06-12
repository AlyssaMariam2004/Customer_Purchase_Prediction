"""
trainer.py

Handles:
- Model retraining logic
- Pickle-based persistence
- Model version cleanup
- Threshold-based triggering of retraining

Depends on external modules for configuration and feature preparation.
"""

import os
import time
import logging
import joblib
import pandas as pd
from datetime import datetime

from config import (
    CSV_PATH, RETRAIN_INTERVAL, ROW_GROWTH_THRESHOLD,
    MODEL_DIR, DF_PATH, MODEL_PATH
)
from recommender import prepare_features

# Global state to track last retrain status
last_retrain_time = time.time()
last_row_count = 0


def save_model(raw_data: pd.DataFrame, processed_data: pd.DataFrame) -> None:
    """
    Saves the raw and processed model data to disk using joblib.

    Includes both:
    - Overwrites to defined paths
    - Timestamped backups for historical versioning

    Parameters:
    ----------
    raw_data : pd.DataFrame
        The raw customer-product dataset.
    processed_data : pd.DataFrame
        The DataFrame output from clustering with features and cluster labels.
    """
    try:
        if not isinstance(processed_data, pd.DataFrame):
            raise ValueError("processed_data must be a pandas DataFrame.")
        if not isinstance(raw_data, pd.DataFrame):
            raise ValueError("raw_data must be a pandas DataFrame.")

        # Save latest versions
        joblib.dump(raw_data, DF_PATH)
        joblib.dump(processed_data, MODEL_PATH)
        logging.info("Model saved to DF_PATH and MODEL_PATH.")

        # Save timestamped versions for backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        joblib.dump(raw_data, os.path.join(MODEL_DIR, f"df_{timestamp}.pkl"))
        joblib.dump(processed_data, os.path.join(MODEL_DIR, f"final_df_{timestamp}.pkl"))

        cleanup_old_pickles()

    except Exception as e:
        logging.error(f"Error while saving models: {e}")


def cleanup_old_pickles(keep: int = 1) -> None:
    """
    Cleans up old pickle backup files to manage disk usage.

    Parameters:
    ----------
    keep : int
        The number of most recent model versions to keep.
    """
    try:
        def is_valid_pickle(fname): return fname.endswith(".pkl") and ("df_" in fname or "final_df_" in fname)

        all_pickles = sorted(
            [f for f in os.listdir(MODEL_DIR) if is_valid_pickle(f)],
            reverse=True
        )

        # We save both df_*.pkl and final_df_*.pkl — remove old pairs beyond keep count
        pickles_to_remove = all_pickles[keep * 2:]

        for file_name in pickles_to_remove:
            os.remove(os.path.join(MODEL_DIR, file_name))
            logging.info(f"Removed old model backup: {file_name}")

    except Exception as e:
        logging.error(f"Error during model cleanup: {e}")


def retrain_model() -> None:
    """
    Loads data from CSV, prepares features, retrains clustering model,
    and persists the updated model state.

    Updates the global last retrain time and row count.
    """
    global last_retrain_time, last_row_count

    try:
        if not os.path.exists(CSV_PATH):
            logging.warning("CSV file not found. Cannot retrain model.")
            return

        df_raw = pd.read_csv(CSV_PATH)
        if df_raw.empty:
            logging.warning("CSV file is empty. Cannot retrain model.")
            return

        processed_df = prepare_features(df_raw)
        if not isinstance(processed_df, pd.DataFrame):
            raise ValueError("prepare_features must return a pandas DataFrame.")

        save_model(df_raw, processed_df)

        last_retrain_time = time.time()
        last_row_count = len(df_raw)

        logging.info("Model retrained successfully.")

    except Exception as e:
        logging.error(f"Error during retrain_model: {e}")


def maybe_retrain_model() -> None:
    """
    Checks if retraining should be triggered based on time or data size threshold.

    Triggers retraining if:
    - Time since last retrain exceeds RETRAIN_INTERVAL
    - Row count growth exceeds ROW_GROWTH_THRESHOLD
    """
    global last_row_count

    try:
        if not os.path.exists(CSV_PATH):
            logging.warning("CSV file not found. Cannot check for retraining.")
            return

        df = pd.read_csv(CSV_PATH)
        if df.empty:
            logging.warning("CSV file is empty. Skipping retraining check.")
            return

        time_exceeded = time.time() - last_retrain_time > RETRAIN_INTERVAL
        growth_exceeded = len(df) - last_row_count > ROW_GROWTH_THRESHOLD

        if time_exceeded or growth_exceeded:
            logging.info("Triggering retraining based on threshold.")
            retrain_model()
        else:
            logging.info("No retraining needed at this time.")

    except Exception as e:
        logging.error(f"Error in maybe_retrain_model: {e}")
