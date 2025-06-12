import pandas as pd
import mysql.connector
import os
import logging
from config import DB_CONFIG, CSV_PATH

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
    logging.info("Starting data sync...")
    df_new = fetch_data()
    logging.info(f"Fetched {len(df_new)} new rows.")

    if os.path.exists(CSV_PATH):
        df_existing = pd.read_csv(CSV_PATH)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates()
    else:
        df_combined = df_new.drop_duplicates()

    df_combined.to_csv(CSV_PATH, index=False)
    logging.info(f"Data synced. Current total: {len(df_combined)} rows.")
