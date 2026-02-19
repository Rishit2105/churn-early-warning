# Customer Churn Early-Warning System

A machine learning pipeline that predicts which customers are likely to cancel 
their subscription, built with Python, SQLite, Groq AI, and scikit-learn.

## What It Does

Takes raw billing data (customers, subscriptions, invoices) and produces a 
churn risk score for every customer — exported as a CSV ready for dashboards.

## Pipeline
```
generate_mock_data.py   → Creates realistic billing data (100 customers)
load_to_sqlite.py       → Loads data into SQLite database
feature_engineering.py  → Builds churn signals + Groq AI risk scores
train_model.py          → Trains Random Forest classifier
score_customers.py      → Scores all customers, exports churn_scores.csv
```

## Tech Stack

- **Python** — data processing and ML
- **SQLite** — storing and querying billing data
- **Groq API (Llama 3.3)** — AI-powered churn risk assessment
- **scikit-learn** — Random Forest classifier
- **pandas** — data manipulation

## Features Used in the Model

| Feature | Why It Matters |
|---------|---------------|
| `payment_failure_rate` | Missed payments = strong churn signal |
| `subscription_age_days` | Longer subscribers are more loyal |
| `total_paid` | High spend = more invested in product |
| `total_invoices` | More billing cycles = more engagement |
| `plan_amount` | Higher plans may have different churn patterns |
| `groq_risk_score` | AI assessment of billing behavior (1-10) |

## Output

`data/churn_scores.csv` — one row per customer with:
- Churn probability (0-100%)
- Risk level (HIGH / MEDIUM / LOW)
- All features used for scoring

## Setup
```bash
pip install -r requirements.txt
```

Add your Groq API key to `config.py`:
```python
GROQ_API_KEY = "your_key_here"
```

Run the pipeline in order:
```bash
python scripts/generate_mock_data.py
python scripts/load_to_sqlite.py
python scripts/feature_engineering.py
python scripts/train_model.py
python scripts/score_customers.py
```

## Project Structure
```
churn-early-warning/
├── data/               # CSV files and SQLite database
├── models/             # Saved ML model
├── scripts/            # One script per pipeline stage
├── config.py           # API keys (not committed)
└── requirements.txt
```

## Design Decisions

- **SQLite over Postgres** — portable, no server needed, same SQL syntax
- **Random Forest over neural networks** — explainable, works on small data
- **Groq API** — adds an LLM layer for behavioral risk assessment
- **One script per stage** — easy to re-run any individual step