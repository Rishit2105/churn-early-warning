import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# ----------------------------------------------------------------
# PART 1: Load the feature table
# ----------------------------------------------------------------
df = pd.read_csv("data/features.csv")
print(f"Loaded {len(df)} customers from features.csv")

# ----------------------------------------------------------------
# PART 2: Prepare features (X) and label (y)
# ----------------------------------------------------------------
# These are the columns the model will learn from
feature_cols = [
    "plan_amount",
    "subscription_age_days",
    "total_invoices",
    "total_paid",
    "payment_failure_rate",
    "groq_risk_score",
]

# Replace -1 groq scores (customers we didn't score) with the average
avg_score = df[df["groq_risk_score"] != -1]["groq_risk_score"].mean()
df["groq_risk_score"] = df["groq_risk_score"].replace(-1, avg_score)

X = df[feature_cols]      # Features
y = df["is_churned"]      # Label: 1 = churned, 0 = active

print(f"\nChurn breakdown:")
print(f"  Active customers:    {(y == 0).sum()}")
print(f"  Churned customers:   {(y == 1).sum()}")

# ----------------------------------------------------------------
# PART 3: Split into train and test sets
# ----------------------------------------------------------------
# 80% of data for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining set: {len(X_train)} customers")
print(f"Testing set:  {len(X_test)} customers")

# ----------------------------------------------------------------
# PART 4: Train the Random Forest model
# ----------------------------------------------------------------
# Random Forest = many decision trees working together
# It's robust, easy to explain, and works well on small datasets
model = RandomForestClassifier(
    n_estimators=100,   # 100 decision trees
    random_state=42,    # So results are reproducible
)

print("\nTraining model...")
model.fit(X_train, y_train)
print("Training complete!")

# ----------------------------------------------------------------
# PART 5: Evaluate the model
# ----------------------------------------------------------------
y_pred = model.predict(X_test)

print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred)*100:.1f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=["Active", "Churned"]))

# Feature importance — which signals matter most?
print("Feature Importance (what the model relies on most):")
importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

for _, row in importance.iterrows():
    bar = "█" * int(row["importance"] * 50)
    print(f"  {row['feature']:<30} {bar} {row['importance']:.3f}")

# ----------------------------------------------------------------
# PART 6: Save the trained model
# ----------------------------------------------------------------
os.makedirs("models", exist_ok=True)
with open("models/churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to models/churn_model.pkl")
print("\nDone!")