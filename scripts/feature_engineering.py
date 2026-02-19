# feature_engineering.py
# Pulls data from SQLite, calculates churn signals per customer,
# and calls Groq API to get an AI risk score for each customer.

import sqlite3
import pandas as pd
import os
import time
from groq import Groq

# Read API key directly from config.py
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "gsk_your_actual_key_here")
print(f"API key loaded: {GROQ_API_KEY[:8]}...")

# ----------------------------------------------------------------
# Connect to database
# ----------------------------------------------------------------
conn = sqlite3.connect("data/churn.db")
print("Connected to database.")

# ----------------------------------------------------------------
# PART 1: Calculate features using SQL + pandas
# ----------------------------------------------------------------
customers_df     = pd.read_sql("SELECT * FROM customers", conn)
subscriptions_df = pd.read_sql("SELECT * FROM subscriptions", conn)
invoices_df      = pd.read_sql("SELECT * FROM invoices", conn)

# Invoice features per customer
invoice_features = invoices_df.groupby("customer_id").agg(
    total_invoices  = ("invoice_id", "count"),
    total_paid      = ("amount_paid", "sum"),
    failed_invoices = ("status", lambda x: (x == "failed").sum()),
).reset_index()

invoice_features["payment_failure_rate"] = (
    invoice_features["failed_invoices"] / invoice_features["total_invoices"]
).round(2)

# Subscription features
sub_features = subscriptions_df[["customer_id", "plan_amount", "created_at", "is_churned"]].copy()
sub_features["created_at"] = pd.to_datetime(sub_features["created_at"])
sub_features["subscription_age_days"] = (pd.Timestamp.now() - sub_features["created_at"]).dt.days

# Merge everything
features_df = customers_df[["customer_id", "name"]].merge(
    sub_features[["customer_id", "plan_amount", "subscription_age_days", "is_churned"]],
    on="customer_id"
).merge(
    invoice_features[["customer_id", "total_invoices", "total_paid", "payment_failure_rate"]],
    on="customer_id"
)

print(f"Built features for {len(features_df)} customers.")

# ----------------------------------------------------------------
# PART 2: Call Groq API for AI risk scores (first 20 customers)
# ----------------------------------------------------------------
client = Groq(api_key=GROQ_API_KEY)

def get_groq_risk_score(row):
    prompt = f"""You are a customer churn analyst. Based on the billing data below,
give a churn risk score from 1 to 10.
1 = very unlikely to churn, 10 = very likely to churn.

Customer data:
- Subscription age: {row['subscription_age_days']} days
- Plan amount: Rs.{row['plan_amount']} per month
- Total invoices: {row['total_invoices']}
- Payment failure rate: {row['payment_failure_rate']*100:.0f}%
- Total amount paid: Rs.{row['total_paid']}

Reply with ONLY a single number between 1 and 10. Nothing else."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
        )
        score = int(response.choices[0].message.content.strip())
        return score
    except Exception as e:
        print(f"  Groq error: {e}")
        return 5

print("\nCalling Groq API for risk scores (first 20 customers)...")
groq_scores = []

for i, row in features_df.head(20).iterrows():
    score = get_groq_risk_score(row)
    groq_scores.append(score)
    print(f"  {row['name']}: risk score = {score}/10")
    time.sleep(1)

remaining = [-1] * (len(features_df) - 20)
features_df["groq_risk_score"] = groq_scores + remaining

# ----------------------------------------------------------------
# PART 3: Save feature table
# ----------------------------------------------------------------
features_df.to_csv("data/features.csv", index=False)
features_df.to_sql("features", conn, if_exists="replace", index=False)
conn.close()

print(f"\nSaved features to data/features.csv")
print(features_df[["name", "payment_failure_rate", "groq_risk_score", "is_churned"]].head(5))
print("\nDone!")