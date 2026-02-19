import pandas as pd
import random
import os
from datetime import datetime, timedelta

# So results are the same every time we run
random.seed(42)

# ----------------------------------------------------------------
# Helper: generate a random date within the last N days
# ----------------------------------------------------------------
def random_date(days_back=365):
    return datetime.now() - timedelta(days=random.randint(0, days_back))

# ----------------------------------------------------------------
# Generate 100 fake customers
# ----------------------------------------------------------------
def generate_customers(n=100):
    first_names = ["Aarav","Priya","Rohan","Sneha","Amit","Divya","Rahul","Pooja",
                   "Vikram","Anjali","Arjun","Neha","Karan","Isha","Siddharth",
                   "Meera","Raj","Kavya","Aditya","Shreya"]
    last_names  = ["Sharma","Patel","Singh","Kumar","Gupta","Joshi","Mehta",
                   "Shah","Verma","Nair","Iyer","Reddy","Das","Chopra","Bose"]

    customers = []
    for i in range(n):
        created = random_date(365)
        customers.append({
            "customer_id": f"cust_{i+1:04d}",
            "name": f"{random.choice(first_names)} {random.choice(last_names)}",
            "email": f"user{i+1}@example.com",
            "created_at": created.strftime("%Y-%m-%d"),
            "city": random.choice(["Mumbai","Delhi","Bangalore","Hyderabad","Chennai","Pune"]),
        })

    return pd.DataFrame(customers)

# ----------------------------------------------------------------
# Generate subscriptions — some active, some cancelled
# ----------------------------------------------------------------
def generate_subscriptions(customers_df):
    subscriptions = []

    for _, customer in customers_df.iterrows():
        created = datetime.strptime(customer["created_at"], "%Y-%m-%d")

        # 25% of customers have cancelled — these are our "churned" ones
        is_churned = random.random() < 0.25
        status = "cancelled" if is_churned else "active"

        # Churned customers had shorter subscriptions
        if is_churned:
            end_date = created + timedelta(days=random.randint(10, 90))
        else:
            end_date = datetime.now() + timedelta(days=random.randint(10, 365))

        subscriptions.append({
            "subscription_id": f"sub_{customer['customer_id']}",
            "customer_id": customer["customer_id"],
            "status": status,
            "plan": random.choice(["basic", "pro", "enterprise"]),
            "plan_amount": random.choice([499, 999, 2999]),  # INR
            "created_at": created.strftime("%Y-%m-%d"),
            "current_period_end": end_date.strftime("%Y-%m-%d"),
            "is_churned": int(is_churned),  # 1 = churned, 0 = active (our ML label)
        })

    return pd.DataFrame(subscriptions)

# ----------------------------------------------------------------
# Generate invoices — some paid, some failed
# ----------------------------------------------------------------
def generate_invoices(customers_df):
    invoices = []
    invoice_id = 1

    for _, customer in customers_df.iterrows():
        created = datetime.strptime(customer["created_at"], "%Y-%m-%d")

        # Each customer has 1-12 invoices
        num_invoices = random.randint(1, 12)

        for j in range(num_invoices):
            invoice_date = created + timedelta(days=j * 30)

            # Some customers have payment failures — strong churn signal
            is_failed = random.random() < 0.2
            status = "failed" if is_failed else "paid"
            amount_paid = 0 if is_failed else random.choice([499, 999, 2999])

            invoices.append({
                "invoice_id": f"inv_{invoice_id:05d}",
                "customer_id": customer["customer_id"],
                "amount_due": random.choice([499, 999, 2999]),
                "amount_paid": amount_paid,
                "status": status,
                "created_at": invoice_date.strftime("%Y-%m-%d"),
            })
            invoice_id += 1

    return pd.DataFrame(invoices)

# ----------------------------------------------------------------
# Main: generate and save everything
# ----------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("Generating customers...")
    customers_df = generate_customers(100)
    customers_df.to_csv("data/customers.csv", index=False)
    print(f"  Saved {len(customers_df)} customers to data/customers.csv")

    print("Generating subscriptions...")
    subscriptions_df = generate_subscriptions(customers_df)
    subscriptions_df.to_csv("data/subscriptions.csv", index=False)
    print(f"  Saved {len(subscriptions_df)} subscriptions to data/subscriptions.csv")

    print("Generating invoices...")
    invoices_df = generate_invoices(customers_df)
    invoices_df.to_csv("data/invoices.csv", index=False)
    print(f"  Saved {len(invoices_df)} invoices to data/invoices.csv")

    print("\nDone! Check your data/ folder.")