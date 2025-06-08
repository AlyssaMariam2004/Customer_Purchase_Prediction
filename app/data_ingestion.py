import pandas as pd
import mysql.connector
import os
from app.config import DB_CONFIG, CSV_PATH
import logging

def fetch_data():
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

def sync_data():
    new_data = fetch_data()

    if not os.path.exists(CSV_PATH):
        logging.warning(f"CSV not found at {CSV_PATH}. Cannot sync. Please provide base CSV.")
        return  # Don't create or overwrite CSV — user must provide it manually.

    old_data = pd.read_csv(CSV_PATH)

    # Drop exact duplicates
    combined = pd.concat([old_data, new_data], ignore_index=True)
    combined.drop_duplicates(subset=["Order ID", "Product ID"], inplace=True)

    if len(combined) > len(old_data):
        combined.to_csv(CSV_PATH, index=False)
        logging.info(f"Appended new data. CSV updated. Total rows: {len(combined)}")
    else:
        logging.info("No new unique rows found. CSV not updated.")

