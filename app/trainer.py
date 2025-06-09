import pandas as pd
import time
import logging
from app.config import CSV_PATH, RETRAIN_INTERVAL, ROW_GROWTH_THRESHOLD
from app.recommender import prepare_features

# Global variables to hold dataframes and retraining state
df = pd.DataFrame()
final_df = pd.DataFrame()
last_retrain_time = time.time()
last_row_count = 0

def retrain_model():
    """
    Retrains the recommendation model using the latest data from the CSV file.

    Steps performed:
    - Reads fresh data from the CSV_PATH.
    - Prepares features by calling `prepare_features`.
    - Updates the timestamp of the last retraining.
    - Updates the count of rows present at the time of retraining.
    - Logs the retraining event.

    This function modifies global variables `df`, `final_df`, `last_retrain_time`, and `last_row_count`.
    """
    global df, final_df, last_retrain_time, last_row_count

    # Load the latest dataset from CSV
    df = pd.read_csv(CSV_PATH)

    # Prepare features and assign to final_df (includes clustering)
    final_df = prepare_features(df)

    # Update last retrain timestamp and row count
    last_retrain_time = time.time()
    last_row_count = len(df)

    logging.info("Model retrained.")

def maybe_retrain_model():
    """
    Checks if conditions for retraining the model are met, and triggers retraining if needed.

    Conditions checked:
    - Whether the time since the last retrain exceeds RETRAIN_INTERVAL.
    - Whether the number of new rows added since last retrain exceeds ROW_GROWTH_THRESHOLD.

    If either condition is true, calls `retrain_model()`.

    Uses the global `last_row_count` and `last_retrain_time` to track the retraining state.
    """
    global last_row_count

    current_time = time.time()
    new_data = pd.read_csv(CSV_PATH)

    # Check if retrain interval elapsed or enough new data has been added
    if (current_time - last_retrain_time > RETRAIN_INTERVAL or
        len(new_data) - last_row_count > ROW_GROWTH_THRESHOLD):
        retrain_model()
