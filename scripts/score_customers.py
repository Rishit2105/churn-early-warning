import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------------------------------
# PART 1: Load the trained model and feature data
# ----------------------------------------------------------------
with open("models/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("data/features.csv")
print(f"Loaded {len(df)} customers to score.")

# ----------------------------------------------------------------
# PART 2: Prepare features (same columns as training)
# ----------------------------------------------------------------
feature_cols = [
    "plan_amount",
    "subscription_age_days",
    "total_invoices",
    "total_paid",
    "payment_failure_rate",
    "groq_risk_score",
]

# Replace -1 groq scores with average (same as during training)
avg_score = df[df["groq_risk_score"] != -1]["groq_risk_score"].mean()
df["groq_risk_score"] = df["groq_risk_score"].replace(-1, avg_score)

X = df[feature_cols]

# ----------------------------------------------------------------
# PART 3: Score every customer
# ----------------------------------------------------------------
# predict_proba gives probability of each class [active, churned]
# We take column [1] which is the probability of churning
churn_probabilities = model.predict_proba(X)[:, 1]

# Convert to percentage and round
df["churn_risk_pct"] = (churn_probabilities * 100).round(1)

# Add a simple risk label
def risk_label(pct):
    if pct >= 70:
        return "HIGH"
    elif pct >= 40:
        return "MEDIUM"
    else:
        return "LOW"

df["risk_level"] = df["churn_risk_pct"].apply(risk_label)

# ----------------------------------------------------------------
# PART 4: Export final CSV
# ----------------------------------------------------------------
output_cols = [
    "customer_id",
    "name",
    "plan_amount",
    "subscription_age_days",
    "payment_failure_rate",
    "groq_risk_score",
    "churn_risk_pct",
    "risk_level",
    "is_churned",
]

output_df = df[output_cols].sort_values("churn_risk_pct", ascending=False)
output_df.to_csv("data/churn_scores.csv", index=False)

# ----------------------------------------------------------------
# PART 5: Print summary
# ----------------------------------------------------------------
print("\nChurn Risk Summary:")
print(f"  HIGH risk customers:   {(output_df['risk_level'] == 'HIGH').sum()}")
print(f"  MEDIUM risk customers: {(output_df['risk_level'] == 'MEDIUM').sum()}")
print(f"  LOW risk customers:    {(output_df['risk_level'] == 'LOW').sum()}")

print("\nTop 10 highest risk customers:")
print(output_df[["name", "churn_risk_pct", "risk_level",
                  "payment_failure_rate", "is_churned"]].head(10).to_string(index=False))

print("\nSaved to data/churn_scores.csv")
print("\nDone!")