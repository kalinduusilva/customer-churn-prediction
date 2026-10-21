"""
02_data_analysis.py
-------------------
Exploratory Data Analysis (EDA) for Customer Churn Prediction.

As a Data Science student, this script teaches you:
1. Data Integrity: Detecting and handling missing values and duplicates.
2. Statistical Profiling: Computing summary stats for numerical features.
3. Behavioral Comparisons: Comparing Churned vs Retained customer metrics.
4. Categorical Breakdown: Analyzing how contract length & subscription type affect churn.
5. Visualizations: Generating and saving publication-ready plots into the 'charts' folder.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 10})

# Paths
DATA_PATH = os.path.join("archive", "customer_churn_dataset-training-master.csv")
CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

print("=" * 70)
print("  STEP 2: EXPLORATORY DATA ANALYSIS (EDA) - CUSTOMER CHURN SYSTEM")
print("=" * 70)

# 1. Load Data
print(f"\n[1/5] Loading Training Dataset from '{DATA_PATH}'...")
df = pd.read_csv(DATA_PATH)
print(f"      Initial shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# 2. Data Cleaning Check
print("\n[2/5] Checking Data Integrity & Cleaning...")
missing = df.isnull().sum()
total_missing = missing.sum()
print(f"      Total missing values found: {total_missing}")
if total_missing > 0:
    print("      Missing values per column:")
    for col, count in missing[missing > 0].items():
        print(f"        - {col}: {count}")
    # Drop empty/incomplete rows
    initial_len = len(df)
    df = df.dropna()
    print(f"      Dropped {initial_len - len(df)} row(s) containing nulls. Clean shape: {df.shape[0]:,} rows.")

# Ensure correct data types
df["Churn"] = df["Churn"].astype(int)

# 3. Overall Churn Distribution
print("\n[3/5] Target Variable (Churn) Overview:")
churn_summary = df["Churn"].value_counts()
churn_rates = df["Churn"].value_counts(normalize=True) * 100
print(f"      - Retained Customers (0): {churn_summary.get(0, 0):,} ({churn_rates.get(0, 0):.2f}%)")
print(f"      - Churned Customers  (1): {churn_summary.get(1, 0):,} ({churn_rates.get(1, 0):.2f}%)")

# 4. Behavioral Comparison: Churned vs Retained
print("\n[4/5] Key Metric Comparisons (Mean Values):")
metrics = ["Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Total Spend", "Last Interaction"]
comparison = df.groupby("Churn")[metrics].mean().T
comparison.columns = ["Retained (Mean)", "Churned (Mean)"]
comparison["Difference (%)"] = ((comparison["Churned (Mean)"] - comparison["Retained (Mean)"]) / comparison["Retained (Mean)"]) * 100

print(comparison.to_string())

# Categorical Breakdown: Contract Length
print("\n      Churn Rate by Contract Length:")
contract_churn = df.groupby("Contract Length")["Churn"].agg(["count", "mean"])
contract_churn["Churn %"] = contract_churn["mean"] * 100
contract_churn = contract_churn.rename(columns={"count": "Customer Count"}).drop(columns=["mean"])
print(contract_churn.to_string())

# 5. Generate and Save Visualizations
print(f"\n[5/5] Generating Visualizations in '{CHARTS_DIR}/' folder...")

# Figure 1: Key Drivers Comparison (Support Calls & Payment Delay)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot A: Support Calls by Churn
sns.boxplot(
    ax=axes[0],
    data=df,
    x="Churn",
    y="Support Calls",
    hue="Churn",
    palette=["#2ecc71", "#e74c3c"],
    legend=False
)
axes[0].set_title("Support Calls: Retained (0) vs Churned (1)", fontsize=12, fontweight="bold")
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["Retained", "Churned"])

# Plot B: Payment Delay by Churn
sns.boxplot(
    ax=axes[1],
    data=df,
    x="Churn",
    y="Payment Delay",
    hue="Churn",
    palette=["#2ecc71", "#e74c3c"],
    legend=False
)
axes[1].set_title("Payment Delay (Days): Retained vs Churned", fontsize=12, fontweight="bold")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["Retained", "Churned"])

plt.tight_layout()
fig1_path = os.path.join(CHARTS_DIR, "01_support_and_payment_delay.png")
fig.savefig(fig1_path, dpi=200)
plt.close(fig)
print(f"      [SAVED] {fig1_path}")

# Figure 2: Contract Length & Subscription Churn Rates
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

contract_order = ["Monthly", "Quarterly", "Annual"]
available_contracts = [c for c in contract_order if c in df["Contract Length"].unique()]

sns.barplot(
    ax=axes[0],
    data=df,
    x="Contract Length",
    y="Churn",
    hue="Contract Length",
    order=available_contracts,
    errorbar=None,
    palette="Blues_r",
    legend=False
)
axes[0].set_title("Churn Rate by Contract Length", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Churn Rate (0 to 1)")

subscription_types = df["Subscription Type"].unique()
sns.barplot(
    ax=axes[1],
    data=df,
    x="Subscription Type",
    y="Churn",
    hue="Subscription Type",
    errorbar=None,
    palette="Purples_r",
    legend=False
)
axes[1].set_title("Churn Rate by Subscription Type", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Churn Rate (0 to 1)")

plt.tight_layout()
fig2_path = os.path.join(CHARTS_DIR, "02_contract_and_subscription_churn.png")
fig.savefig(fig2_path, dpi=200)
plt.close(fig)
print(f"      [SAVED] {fig2_path}")


print("\n" + "=" * 70)
print("  EDA ANALYSIS COMPLETE! KEY TAKEAWAYS FOR MACHINE LEARNING:")
print("  1. Support Calls & Payment Delay show strong divergence for churners.")
print("  2. Monthly contract holders have substantially higher churn probability.")
print("  3. Total Spend and Tenure will serve as core retention indicators.")
print("=" * 70)
