"""
03_preprocessing.py
-------------------
Data Preprocessing & Feature Engineering for Customer Churn Prediction.

As a Data Science student, this script teaches you:
1. Leakage Prevention: Fitting transformers ONLY on training data, then transforming test data.
2. Feature Engineering: Creating domain-driven features (e.g. Spend per Tenure, Support Intensity).
3. Scikit-Learn Pipelines: Using ColumnTransformer for combined numeric scaling & categorical one-hot encoding.
4. Artifact Persistence: Saving the fitted preprocessor with joblib for model training and Streamlit deployment.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define directories
TRAIN_RAW_PATH = os.path.join("archive", "customer_churn_dataset-training-master.csv")
TEST_RAW_PATH = os.path.join("archive", "customer_churn_dataset-testing-master.csv")
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 70)
print("  STEP 3: DATA PREPROCESSING & FEATURE ENGINEERING")
print("=" * 70)

# 1. Load Datasets
print(f"\n[1/6] Loading raw datasets...")
df_train = pd.read_csv(TRAIN_RAW_PATH)
df_test = pd.read_csv(TEST_RAW_PATH)

print(f"      Train raw shape: {df_train.shape[0]:,} rows x {df_train.shape[1]} columns")
print(f"      Test raw shape : {df_test.shape[0]:,} rows x {df_test.shape[1]} columns")

# 2. Data Cleaning
print("\n[2/6] Cleaning missing values and removing identifier columns...")
# Drop null rows in training set (1 trailing row)
df_train = df_train.dropna().reset_index(drop=True)
df_test = df_test.dropna().reset_index(drop=True)

# Separate Target (y) and Features (X)
y_train = df_train["Churn"].astype(int).values
y_test = df_test["Churn"].astype(int).values

X_train = df_train.drop(columns=["CustomerID", "Churn"])
X_test = df_test.drop(columns=["CustomerID", "Churn"])

print(f"      Removed 'CustomerID' (identifier) and extracted 'Churn' target.")
print(f"      Clean Train: {X_train.shape[0]:,} rows | Clean Test: {X_test.shape[0]:,} rows")

# 3. Feature Engineering
print("\n[3/6] Engineering new domain-specific features...")

def add_engineered_features(df_features: pd.DataFrame) -> pd.DataFrame:
    df_out = df_features.copy()
    # Feature 1: Spend per month of tenure (Value velocity)
    df_out["Spend_per_Tenure"] = df_out["Total Spend"] / (df_out["Tenure"] + 1)
    
    # Feature 2: Support intensity (Calls per unit of usage frequency)
    df_out["Support_Intensity"] = df_out["Support Calls"] / (df_out["Usage Frequency"] + 1)
    
    # Feature 3: Payment Delay Risk Flag (Payment delay > 15 days indicates high friction)
    df_out["High_Delay_Flag"] = (df_out["Payment Delay"] > 15).astype(int)
    
    return df_out

X_train_eng = add_engineered_features(X_train)
X_test_eng = add_engineered_features(X_test)

print(f"      Added 3 engineered features:")
print("        - 'Spend_per_Tenure'   : Total Spend / (Tenure + 1)")
print("        - 'Support_Intensity'  : Support Calls / (Usage Frequency + 1)")
print("        - 'High_Delay_Flag'    : Binary indicator (Payment Delay > 15 days)")
print(f"      New feature count: {X_train_eng.shape[1]}")

# 4. Construct Preprocessing Pipeline
print("\n[4/6] Setting up Scikit-Learn ColumnTransformer...")

categorical_cols = ["Gender", "Subscription Type", "Contract Length"]
numerical_cols = [col for col in X_train_eng.columns if col not in categorical_cols]

print(f"      Numerical features ({len(numerical_cols)}): {numerical_cols}")
print(f"      Categorical features ({len(categorical_cols)}): {categorical_cols}")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols)
    ]
)

# 5. Fit Preprocessor on Training Data & Transform Both
print("\n[5/6] Fitting preprocessor on Train and transforming Test data...")
X_train_processed = preprocessor.fit_transform(X_train_eng)
X_test_processed = preprocessor.transform(X_test_eng)

# Extract transformed feature names
encoded_cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
all_feature_names = numerical_cols + list(encoded_cat_names)

print(f"      Transformation complete:")
print(f"      - X_train processed shape: {X_train_processed.shape}")
print(f"      - X_test processed shape : {X_test_processed.shape}")
print(f"      - Total transformed features: {len(all_feature_names)}")

# 6. Save Artifacts for Model Training and Streamlit App
print("\n[6/6] Saving preprocessor pipeline and processed data arrays...")

# Save pipeline object
preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
joblib.dump(preprocessor, preprocessor_path)
print(f"      [SAVED] Preprocessor pipeline: {preprocessor_path}")

# Save processed arrays & metadata into compressed .npz archive
data_archive_path = os.path.join(PROCESSED_DIR, "churn_processed_data.npz")
np.savez_compressed(
    data_archive_path,
    X_train=X_train_processed,
    y_train=y_train,
    X_test=X_test_processed,
    y_test=y_test,
    feature_names=all_feature_names
)
print(f"      [SAVED] Processed data archive: {data_archive_path}")

print("\n" + "=" * 70)
print("  STEP 3 COMPLETE! READY FOR MODEL BUILDING & EVALUATION")
print(f"  Feature set ({len(all_feature_names)} features):")
for idx, name in enumerate(all_feature_names, start=1):
    print(f"    {idx:2d}. {name}")
print("=" * 70)
