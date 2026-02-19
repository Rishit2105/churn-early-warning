# Step 3: Load data into SQLite database

import pandas as pd
import sqlite3
import os

# ----------------------------------------------------------------
# Connect to SQLite (creates the file if it doesn't exist)
# ----------------------------------------------------------------
conn = sqlite3.connect("data/churn.db")
print("Connected to SQLite database: data/churn.db")

# ----------------------------------------------------------------
# Load each CSV into a database table
# ----------------------------------------------------------------

# Customers table
customers_df = pd.read_csv("data/customers.csv")
customers_df.to_sql("customers", conn, if_exists="replace", index=False)
print(f"Loaded {len(customers_df)} rows into 'customers' table")

# Subscriptions table
subscriptions_df = pd.read_csv("data/subscriptions.csv")
subscriptions_df.to_sql("subscriptions", conn, if_exists="replace", index=False)
print(f"Loaded {len(subscriptions_df)} rows into 'subscriptions' table")

# Invoices table
invoices_df = pd.read_csv("data/invoices.csv")
invoices_df.to_sql("invoices", conn, if_exists="replace", index=False)
print(f"Loaded {len(invoices_df)} rows into 'invoices' table")

# ----------------------------------------------------------------
# Verify it worked by running a quick SQL query
# ----------------------------------------------------------------
print("\nVerifying data with SQL queries:")

result = pd.read_sql("SELECT COUNT(*) as total_customers FROM customers", conn)
print(f"  Total customers: {result['total_customers'][0]}")

result = pd.read_sql("SELECT status, COUNT(*) as count FROM subscriptions GROUP BY status", conn)
print(f"  Subscription breakdown:\n{result}")

result = pd.read_sql("SELECT status, COUNT(*) as count FROM invoices GROUP BY status", conn)
print(f"  Invoice breakdown:\n{result}")

conn.close()
print("\nDone! Database saved to data/churn.db")