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
    logging.info(f"Fetched {len(new_data)} rows from DB")

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    if os.path.exists(CSV_PATH):
        old_data = pd.read_csv(CSV_PATH)
        logging.info(f"Existing CSV has {len(old_data)} rows")

        combined = pd.concat([old_data, new_data], ignore_index=True)
        combined.drop_duplicates(inplace=True)  # safest for now

        logging.info(f"After deduplication, CSV would have {len(combined)} rows")

        if len(combined) > len(old_data):
            combined.to_csv(CSV_PATH, index=False)
            logging.info("Appended new data and saved CSV.")
        else:
            logging.info("No new rows to add.")
    else:
        new_data.to_csv(CSV_PATH, index=False)
        logging.info("Created CSV for the first time.")
