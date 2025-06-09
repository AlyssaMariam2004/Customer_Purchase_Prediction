"""
Data synchronization module.

This script connects to a MySQL database, fetches new order-related data,
and updates a local CSV file used for model retraining. It avoids duplicates
by checking key columns and logs the sync process.
"""

import pandas as pd
import mysql.connector
import os
from app.config import DB_CONFIG, CSV_PATH
import logging

def fetch_data():
    """
    Connect to the MySQL database and fetch order, customer, and product data.

    Returns:
        pd.DataFrame: A DataFrame containing the selected fields from joined tables.
    """
    conn = mysql.connector.connect(**DB_CONFIG)
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
    df = pd.read_sql(query, conn)
    conn.close()
    return df

logging.basicConfig(level=logging.DEBUG)

def sync_data():
    """
    Fetches new data from the source, appends it to existing CSV data,
    removes duplicates, and saves back to CSV_PATH safely.
    """
    logging.debug(f"Starting sync_data. CSV_PATH={CSV_PATH}")

    # Fetch new data using the fetch_data() function
    df_new = fetch_data()  # your existing fetch_data function that returns a DataFrame
    logging.debug(f"Fetched {len(df_new)} new rows from source.")

    # Check if the CSV file already exists
    if os.path.exists(CSV_PATH):
        logging.debug(f"Existing CSV found at {CSV_PATH}. Loading existing data.")
        # Load existing data from CSV
        df_existing = pd.read_csv(CSV_PATH)
        logging.debug(f"Loaded {len(df_existing)} rows from existing CSV.")

        # Concatenate existing data with new data, ignoring the original index
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        logging.debug("Appended new data to existing data.")

        # Remove duplicate rows to maintain data integrity
        df_combined.drop_duplicates(inplace=True)
        logging.debug(f"Combined dataframe has {len(df_combined)} rows after deduplication.")
    else:
        logging.debug(f"No existing CSV found at {CSV_PATH}. Using new data only.")
        # If no existing CSV, start with new data only, dropping duplicates just in case
        df_combined = df_new.drop_duplicates()
        logging.debug(f"New data has {len(df_combined)} unique rows.")

    # Save the combined and cleaned data back to CSV file
    df_combined.to_csv(CSV_PATH, index=False)
    logging.debug(f"CSV at {CSV_PATH} updated successfully.")

