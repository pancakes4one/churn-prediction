import random
import csv
import os
import sys

def generate_churn_data(n_rows, output_path, seed=42):
    """Generate a synthetic customer churn dataset."""
    random.seed(seed)

    header = [
        "customer_id", "age", "gender", "tenure_months",
        "monthly_charges", "total_charges", "contract_type",
        "payment_method", "num_support_tickets", "churned"
    ]

    contract_types = ["month-to-month", "one-year", "two-year"]
    payment_methods = ["credit_card", "bank_transfer", "electronic_check", "mailed_check"]

    rows = []
    for i in range(n_rows):
        tenure = random.randint(1, 72)
        monthly = round(random.uniform(20, 120), 2)
        contract = random.choice(contract_types)

        # Make churn more likely for month-to-month, high charges, low tenure
        churn_prob = 0.15
        if contract == "month-to-month":
            churn_prob += 0.15
        if monthly > 80:
            churn_prob += 0.10
        if tenure < 12:
            churn_prob += 0.10

        churned = 1 if random.random() < churn_prob else 0

        row = [
            f"CUST-{i+1:05d}",
            random.randint(18, 75),
            random.choice(["M", "F"]),
            tenure,
            monthly,
            round(monthly * tenure + random.uniform(-50, 50), 2),
            contract,
            random.choice(payment_methods),
            random.randint(0, 10),
            churned
        ]
        rows.append(row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    churn_count = sum(r[-1] for r in rows)
    print(f"Generated {n_rows} rows at {output_path}")
    print(f"Churn rate: {churn_count/n_rows:.1%} ({churn_count} churned)")

if __name__ == "__main__":
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/raw/customers.csv"
    generate_churn_data(n_rows, output_path)
Now let us also create a simple training script that reads this data and trains a model. Create src/train.py:

import csv
import os
import sys
import random
import pickle
import json
from collections import Counter

def load_data(path):
    """Load CSV data and return features and labels."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def simple_train_test_split(rows, test_ratio=0.2, seed=42):
    """Split data into train and test sets."""
    random.seed(seed)
    random.shuffle(rows)
    split_idx = int(len(rows) * (1 - test_ratio))
    return rows[:split_idx], rows[split_idx:]

def train_simple_model(train_rows):
    """
    Train a simple rule-based model.
    (In a real project, this would be sklearn or similar.)
    """
    # Learn average churn rate by contract type
    churn_rates = {}
    for contract_type in ["month-to-month", "one-year", "two-year"]:
        matching = [r for r in train_rows if r["contract_type"] == contract_type]
        if matching:
            churned = sum(1 for r in matching if r["churned"] == "1")
            churn_rates[contract_type] = churned / len(matching)
        else:
            churn_rates[contract_type] = 0.5

    # Learn tenure threshold
    churned_tenure = [int(r["tenure_months"]) for r in train_rows if r["churned"] == "1"]
    avg_churned_tenure = sum(churned_tenure) / len(churned_tenure) if churned_tenure else 12

    model = {
        "churn_rates_by_contract": churn_rates,
        "tenure_threshold": avg_churned_tenure,
        "charge_threshold": 80.0
    }
    return model

def predict(model, row):
    """Predict churn for a single row."""
    score = model["churn_rates_by_contract"].get(row["contract_type"], 0.5)
    if int(row["tenure_months"]) < model["tenure_threshold"]:
        score += 0.1
    if float(row["monthly_charges"]) > model["charge_threshold"]:
        score += 0.1
    return 1 if score > 0.4 else 0

def evaluate(model, test_rows):
    """Compute accuracy and other basic metrics."""
    correct = 0
    true_pos = 0
    false_pos = 0
    false_neg = 0

    for row in test_rows:
        pred = predict(model, row)
        actual = int(row["churned"])
        if pred == actual:
            correct += 1
        if pred == 1 and actual == 1:
            true_pos += 1
        if pred == 1 and actual == 0:
            false_pos += 1
        if pred == 0 and actual == 1:
            false_neg += 1

    accuracy = correct / len(test_rows)
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "test_size": len(test_rows)
    }

if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/customers.csv"

    print(f"Loading data from {data_path}...")
    rows = load_data(data_path)
    print(f"Loaded {len(rows)} rows")

    train_rows, test_rows = simple_train_test_split(rows)
    print(f"Train: {len(train_rows)} rows, Test: {len(test_rows)} rows")

    print("Training model...")
    model = train_simple_model(train_rows)

    print("Evaluating...")
    metrics = evaluate(model, test_rows)
    print(f"Accuracy:  {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall:    {metrics['recall']}")

    # Save model
    os.makedirs("models", exist_ok=True)
    with open("models/churn_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to models/churn_model.pkl")

    # Save metrics
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved to metrics/results.json")