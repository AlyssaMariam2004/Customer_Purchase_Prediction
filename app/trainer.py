import pandas as pd
import time
import logging
from app.config import CSV_PATH, RETRAIN_INTERVAL, ROW_GROWTH_THRESHOLD
from app.recommender import prepare_features

df = pd.DataFrame()
final_df = pd.DataFrame()
last_retrain_time = time.time()
last_row_count = 0

def retrain_model():
    global df, final_df, last_retrain_time, last_row_count
    df = pd.read_csv(CSV_PATH) # Read fresh data from CSV
    final_df = prepare_features(df) # Process data and retrain model
    last_retrain_time = time.time() # Update retrain timestamp
    last_row_count = len(df) # Update row count tracker
    logging.info("Model retrained.")  # Log the retraining event

def maybe_retrain_model():
    global last_row_count
    current_time = time.time() # Current timestamp
    new_data = pd.read_csv(CSV_PATH) # Load current data from CSV

    # Check if enough time has passed OR enough new data has been added since last retrain
    if (current_time - last_retrain_time > RETRAIN_INTERVAL or
        len(new_data) - last_row_count > ROW_GROWTH_THRESHOLD):
        retrain_model()
