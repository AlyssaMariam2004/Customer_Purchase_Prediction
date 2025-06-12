import pandas as pd
import mysql.connector
import os
import logging
from config import DB_CONFIG, CSV_PATH

def fetch_data():
    try:
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
    except mysql.connector.Error as e:
        logging.error(f"MySQL error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during fetch_data: {e}")
    return pd.DataFrame()

def sync_data():
    logging.info("Starting data sync...")
    df_new = fetch_data()

    if df_new.empty:
        logging.warning("No data fetched from the database.")
        return

    try:
        if os.path.exists(CSV_PATH):
            df_existing = pd.read_csv(CSV_PATH)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates()
            logging.info(f"Merged dataset now has {len(df_combined)} rows.")
        else:
            df_combined = df_new.drop_duplicates()
            logging.info("No existing data found. Using fresh data.")

        df_combined.to_csv(CSV_PATH, index=False)
        logging.info("Data sync complete and written to CSV.")
    except Exception as e:
        logging.error(f"Error during sync_data write: {e}")

