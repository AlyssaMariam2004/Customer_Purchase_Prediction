"""
data_ingestion.py

This module handles data extraction from the MySQL database and 
synchronizes it with a local CSV file. It includes functionality to 
fetch fresh data and append it to the existing dataset while ensuring
deduplication.
"""

import os
import logging
import pandas as pd
import mysql.connector

from scheduler.config import DB_CONFIG, CSV_PATH

def fetch_data() -> pd.DataFrame:
    """
    Fetches data from the MySQL database using predefined joins and projections.

    Returns:
        pd.DataFrame: Fetched data containing orders, customers, products, and quantities.
                      Returns an empty DataFrame if fetching fails.
    """
    try:
        # Establish DB connection using config credentials
        conn = mysql.connector.connect(**DB_CONFIG)

        # SQL query to join relevant tables and select necessary fields
        query = """
        SELECT 
            o.OrderID AS `Order ID`,
            c.CustomerID AS `Customer ID`,
            p.WarehouseID AS `Warehouse ID`,
            c.CustomerAge AS `Customer Age`,
            c.Gender AS `Customer Gender`,
            o.OrderDate AS `Date`,
            p.ProductID AS `Product ID`,
            p.SKUID AS `SKU ID`,
            p.Category AS `Category`,
            oi.Quantity AS `Quantity`,
            p.PricePerUnit AS `Price per Unit`
        FROM OrderItems oi
        JOIN Orders o ON oi.OrderID = o.OrderID
        JOIN Customers c ON o.CustomerID = c.CustomerID
        JOIN Products p ON oi.SKUID = p.SKUID;
        """

        # Read data into a DataFrame
        df = pd.read_sql(query, conn)
        conn.close()
        return df

    except mysql.connector.Error as db_err:
        logging.error(f"MySQL connection error: {db_err}")
    except Exception as err:
        logging.error(f"Unexpected error during fetch_data: {err}")
    
    return pd.DataFrame()  # Return empty DataFrame on failure


def sync_data():
    """
    Syncs the latest database records with the local CSV.

    - Fetches new data from the database.
    - Merges it with the existing CSV (if present).
    - Removes duplicates.
    - Writes the final dataset back to the CSV.

    Logs all operations and errors encountered during the sync process.
    """
    logging.info("Starting data sync from database to CSV...")

    # Step 1: Fetch new data
    df_new = fetch_data()

    if df_new.empty:
        logging.warning("No data fetched from the database.")
        return

    try:
        # Step 2: Load existing data if CSV exists
        if os.path.exists(CSV_PATH):
            df_existing = pd.read_csv(CSV_PATH)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates()
            logging.info(f"Merged dataset now contains {len(df_combined)} rows.")
        else:
            df_combined = df_new.drop_duplicates()
            logging.info("No existing CSV found. Creating a new dataset from fetched data.")

        # Step 3: Save combined dataset to CSV
        df_combined.to_csv(CSV_PATH, index=False)
        logging.info("Data successfully written to CSV.")

    except Exception as write_err:
        logging.error(f"Error during CSV write in sync_data: {write_err}")
