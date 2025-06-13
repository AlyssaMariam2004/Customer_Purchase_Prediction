Customer Purchase Prediction Project

This project is divided into two major components:

1. app/ – FastAPI Application for Serving Recommendations :

The FastAPI app serves real-time customer product recommendations based on the most recently trained model. It is fully decoupled from the background scheduler.

Responsibilities:
Loads the latest pickled model (df.pkl, final_df.pkl) during startup.
Accepts customer request data via POST endpoints.
Uses MinMax scaling, KMeans cluster mapping, and cosine similarity to generate top-N product recommendations.
Returns results as JSON with relevant cluster and similarity info.
Logs each step for traceability and error debugging.

The app performs read-only inference using the existing model and does not retrigger any retraining or CSV sync.



2. scheduler/ – Background Engine for Data Syncing & Model Retraining :

The scheduler runs independently from the FastAPI app and manages data ingestion, conditional model retraining, and model persistence in the background.

Responsibilities:
Reads customer transaction data from the MySQL database and appends it to the local data/NewData1.csv file (without overwriting).

Checks whether retraining is needed based on:
    Time elapsed since last training (RETRAIN_INTERVAL)
    Growth in number of rows (ROW_GROWTH_THRESHOLD)

If triggered, performs:
    Feature engineering and KMeans clustering with silhouette score optimization.
    Model pickling for df.pkl (raw data) and final_df.pkl (processed + cluster labels).
    Backup with timestamped versions and auto-cleanup of older pickles.

