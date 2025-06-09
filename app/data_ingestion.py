import pandas as pd
import mysql.connector
import os
from app.config import DB_CONFIG, CSV_PATH
import logging

#make the connection to DB and fetch only required information
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

#appending new data from the DB to the CSV for model retraining
def sync_data():
    new_data = fetch_data()
    logging.info(f"Fetched {len(new_data)} rows from DB") #logs number of rows of data from DB

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True) #checks if the csv exsists

    dedup_cols = ["Order ID", "Product ID", "SKU ID"] 

    if os.path.exists(CSV_PATH): #if the csv is found
        old_data = pd.read_csv(CSV_PATH) #reads data from exsisting data from csv
        logging.info(f"Existing CSV has {len(old_data)} rows") #logs number of existing csv rows 

        # Check how many rows in new_data are truly new and merges only those
        new_unique = new_data.merge(old_data[dedup_cols], on=dedup_cols, how='left', indicator=True)
        new_only = new_unique[new_unique['_merge'] == 'left_only'] #looks for duplicates
        logging.info(f"New unique rows to add: {len(new_only)}") #logs the number of new rows added

        if len(new_only) == 0: #if no new rows then exit the funtion
            logging.info("No new unique rows found in DB. CSV not updated.")
            return
        
        #append the new unique data to the csv
        combined = pd.concat([old_data, new_only[new_only.columns.difference(['_merge'])]], ignore_index=True)
        #logs the new number of rows
        before_dedup = len(combined)
        combined.drop_duplicates(subset=dedup_cols, inplace=True)
        logging.info(f"Dropped {before_dedup - len(combined)} duplicates after concatenation. Final row count: {len(combined)}")
        combined.to_csv(CSV_PATH, index=False)
        logging.info("Appended new unique data and saved CSV.")
    else:
        new_data.to_csv(CSV_PATH, index=False) 
        logging.info("Created CSV for the first time.")
