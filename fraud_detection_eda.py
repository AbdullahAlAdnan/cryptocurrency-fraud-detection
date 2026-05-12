import numpy as np
import pandas as pd

# Load data
df = pd.read_csv(r'C:\Users\abdul\Downloads\train_transaction.csv', nrows=10000)
amounts = df['TransactionAmt'].values
labels = df['isFraud'].values

# Step 1 — Basic info
print("=" * 50)
print("STEP 1 — DATASET OVERVIEW")
print("=" * 50)
print(f"Shape: {amounts.shape}")
print(f"Total transactions: {len(amounts)}")
print(f"Fraud transactions: {np.sum(labels == 1)}")
print(f"Normal transactions: {np.sum(labels == 0)}")
print(f"Fraud rate: {np.mean(labels)*100:.2f}%")

# Step 2 — Basic statistics
print("\n" + "=" * 50)
print("STEP 2 — TRANSACTION AMOUNT STATISTICS")
print("=" * 50)
print(f"Mean amount:   ${np.mean(amounts):.2f}")
print(f"Std amount:    ${np.std(amounts):.2f}")
print(f"Min amount:    ${np.min(amounts):.2f}")
print(f"Max amount:    ${np.max(amounts):.2f}")

# Step 3 — Fraud vs normal
print("\n" + "=" * 50)
print("STEP 3 — FRAUD VS NORMAL COMPARISON")
print("=" * 50)
fraud_amounts = amounts[labels == 1]
normal_amounts = amounts[labels == 0]
print(f"Mean fraud amount:   ${np.mean(fraud_amounts):.2f}")
print(f"Mean normal amount:  ${np.mean(normal_amounts):.2f}")
print(f"Max fraud amount:    ${np.max(fraud_amounts):.2f}")
print(f"Max normal amount:   ${np.max(normal_amounts):.2f}")

# Step 4 — Anomaly detection
print("\n" + "=" * 50)
print("STEP 4 — ANOMALY DETECTION")
print("=" * 50)
mean = np.mean(amounts)
std = np.std(amounts)
threshold = mean + 3 * std
suspicious = amounts[amounts > threshold]
caught = np.sum((amounts > threshold) & (labels == 1))
total_fraud = np.sum(labels == 1)
print(f"Anomaly threshold (3*std): ${threshold:.2f}")
print(f"Suspicious transactions:   {len(suspicious)}")
print(f"Fraud caught:              {caught}/{total_fraud}")

# Task 2 — Different thresholds
print("\n" + "=" * 50)
print("TASK 2 — COMPARING THRESHOLDS")
print("=" * 50)
for multiplier in [2, 3, 4]:
    threshold = mean + multiplier * std
    caught = np.sum((amounts > threshold) & (labels == 1))
    flagged = np.sum(amounts > threshold)
    print(f"{multiplier}*std → threshold ${threshold:.2f} | "
          f"fraud caught {caught}/{total_fraud} | "
          f"total flagged {flagged}")

# Task 3 — Top 10 largest transactions
print("\n" + "=" * 50)
print("TASK 3 — TOP 10 LARGEST TRANSACTIONS")
print("=" * 50)
top10 = np.sort(amounts)[-10:]
for i, amount in enumerate(top10[::-1], 1):
    print(f"#{i}: ${amount:.2f}")

# Task 4 — Percentage above $1000
print("\n" + "=" * 50)
print("TASK 4 — TRANSACTIONS ABOVE $1000")
print("=" * 50)
above_1000 = np.sum(amounts > 1000)
percentage = (above_1000 / len(amounts)) * 100
print(f"Count: {above_1000} transactions")
print(f"Percentage: {percentage:.2f}%")

print("\n" + "=" * 50)
print("ANALYSIS COMPLETE")
print("=" * 50)
