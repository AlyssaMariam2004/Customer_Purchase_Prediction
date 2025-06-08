import pandas as pd
import mysql.connector
import os
from app.config import DB_CONFIG, CSV_PATH
import logging

#Connecting to the DB to get necessary data
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

#handles merging data to the CSV
def sync_data():
    new_data = fetch_data()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    if os.path.exists(CSV_PATH):
        old = pd.read_csv(CSV_PATH) #reads exisiting data
        combined = pd.concat([old, new_data]).drop_duplicates() #drops any duplicates
        combined.to_csv(CSV_PATH, index=False) #appends
        logging.info(f"CSV updated, total rows: {len(combined)}") #logs total rows
    else:
        new_data.to_csv(CSV_PATH, index=False)
        logging.info("CSV created.")

