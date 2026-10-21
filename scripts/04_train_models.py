"""
04_train_models.py
-------------------
Machine Learning Model Training, Evaluation, and Comparison for Customer Churn.

As a Data Science student, this script teaches you:
1. Multi-Model Benchmarking: Training diverse algorithms (Logistic Regression,
   Decision Tree, Random Forest, and Gradient Boosting).
2. Generalization & Validation: Creating a stratified validation split from the training
   distribution to measure in-distribution performance before evaluating external test data.
3. Multi-Metric Evaluation: Evaluating beyond Accuracy (Precision, Recall, F1-Score,
   ROC-AUC) to balance false positives vs false negatives in churn retention.
4. Model Explainability: Extracting feature importances to identify key churn drivers.
5. Model Persistence: Serializing the champion model into 'models/best_model.joblib'
   and recording benchmark results for the upcoming Streamlit application.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ---------------------------------------------------------
# Configuration & Directories
# ---------------------------------------------------------
PROCESSED_DATA_PATH = os.path.join("data", "processed", "churn_processed_data.npz")
MODELS_DIR = "models"
CHARTS_DIR = "charts"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# Visual styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 10})

print("=" * 75)
print("  STEP 4: MODEL TRAINING, BENCHMARKING & CHAMPION SELECTION")
print("=" * 75)

# ---------------------------------------------------------
# [1/6] Load Processed Datasets
# ---------------------------------------------------------
print(f"\n[1/6] Loading preprocessed data from '{PROCESSED_DATA_PATH}'...")
if not os.path.exists(PROCESSED_DATA_PATH):
    raise FileNotFoundError(
        f"Processed data file not found at {PROCESSED_DATA_PATH}. "
        "Please run 'scripts/03_preprocessing.py' first."
    )

data = np.load(PROCESSED_DATA_PATH, allow_pickle=True)
X_train_full = data["X_train"]
y_train_full = data["y_train"]
X_test_ext = data["X_test"]
y_test_ext = data["y_test"]
feature_names = [str(f) for f in data["feature_names"]]

print(f"      Training set: {X_train_full.shape[0]:,} samples x {X_train_full.shape[1]} features")
print(f"      External test set: {X_test_ext.shape[0]:,} samples x {X_test_ext.shape[1]} features")
print(f"      Feature count: {len(feature_names)}")

# ---------------------------------------------------------
# [2/6] Stratified Train-Validation Split
# ---------------------------------------------------------
print("\n[2/6] Creating stratified Train (80%) and Validation (20%) split...")
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=42,
    stratify=y_train_full
)

print(f"      Train split: {X_train.shape[0]:,} samples (Churn: {np.mean(y_train)*100:.1f}%)")
print(f"      Val split  : {X_val.shape[0]:,} samples (Churn: {np.mean(y_val)*100:.1f}%)")
print(f"      Holdout ext: {X_test_ext.shape[0]:,} samples (Churn: {np.mean(y_test_ext)*100:.1f}%)")

# ---------------------------------------------------------
# [3/6] Define Candidate Models
# ---------------------------------------------------------
print("\n[3/6] Initializing candidate models...")
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        random_state=42
    )
}

# ---------------------------------------------------------
# [4/6] Train and Evaluate All Models
# ---------------------------------------------------------
print("\n[4/6] Training and evaluating candidate models...")

results_list = []
trained_models = {}
eval_details = {}

def calculate_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

for name, model in models.items():
    print(f"\n      --> Training '{name}'...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    trained_models[name] = model

    # Validation predictions (In-Distribution Generalization)
    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:, 1]
    val_metrics = calculate_metrics(y_val, val_pred, val_prob)

    # External Test predictions (External Generalization)
    ext_pred = model.predict(X_test_ext)
    ext_prob = model.predict_proba(X_test_ext)[:, 1]
    ext_metrics = calculate_metrics(y_test_ext, ext_pred, ext_prob)

    eval_details[name] = {
        "val": {"preds": val_pred, "probs": val_prob, "metrics": val_metrics},
        "ext": {"preds": ext_pred, "probs": ext_prob, "metrics": ext_metrics},
        "train_time_sec": round(train_time, 2)
    }

    results_list.append({
        "Model": name,
        "Train Time (s)": round(train_time, 2),
        "Val Accuracy": val_metrics["accuracy"],
        "Val Precision": val_metrics["precision"],
        "Val Recall": val_metrics["recall"],
        "Val F1": val_metrics["f1"],
        "Val ROC-AUC": val_metrics["roc_auc"],
        "Ext Test F1": ext_metrics["f1"],
        "Ext Test ROC-AUC": ext_metrics["roc_auc"]
    })

    print(f"          Trained in {train_time:.2f}s | Val F1: {val_metrics['f1']:.4f} | Val ROC-AUC: {val_metrics['roc_auc']:.4f}")

# Convert to DataFrame for readable display
df_results = pd.DataFrame(results_list)
print("\n" + "-" * 75)
print("  MODEL BENCHMARK PERFORMANCE COMPARISON (Validation Set):")
print("-" * 75)
cols_to_print = ["Model", "Train Time (s)", "Val Accuracy", "Val Precision", "Val Recall", "Val F1", "Val ROC-AUC"]
print(df_results[cols_to_print].to_string(index=False, formatters={
    "Train Time (s)": "{:.2f}".format,
    "Val Accuracy": "{:.4f}".format,
    "Val Precision": "{:.4f}".format,
    "Val Recall": "{:.4f}".format,
    "Val F1": "{:.4f}".format,
    "Val ROC-AUC": "{:.4f}".format
}))

# ---------------------------------------------------------
# [5/6] Generate Visualizations
# ---------------------------------------------------------
print(f"\n[5/6] Generating and saving evaluation charts in '{CHARTS_DIR}/'...")

# Figure 03: Performance Metric Comparison Across Models
fig, ax = plt.subplots(figsize=(10, 6))
metrics_plot_df = df_results.melt(
    id_vars=["Model"],
    value_vars=["Val Accuracy", "Val Precision", "Val Recall", "Val F1", "Val ROC-AUC"],
    var_name="Metric",
    value_name="Score"
)
# Clean up metric labels for display
metrics_plot_df["Metric"] = metrics_plot_df["Metric"].str.replace("Val ", "")

sns.barplot(
    data=metrics_plot_df,
    x="Metric",
    y="Score",
    hue="Model",
    palette="viridis",
    ax=ax
)
ax.set_ylim(0.4, 1.05)
ax.set_title("Candidate Models Benchmark (Validation Set)", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Score (0.0 to 1.0)")
ax.set_xlabel("Evaluation Metric")
ax.legend(title="Algorithm", loc="lower right", frameon=True)
plt.tight_layout()
chart_metrics_path = os.path.join(CHARTS_DIR, "03_model_metrics_comparison.png")
plt.savefig(chart_metrics_path, dpi=300)
plt.close()
print(f"      [SAVED] Metrics comparison chart: {chart_metrics_path}")

# Figure 04: Confusion Matrices (2x2 grid for 4 models)
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
axes = axes.flatten()

for idx, (name, details) in enumerate(eval_details.items()):
    cm = confusion_matrix(y_val, details["val"]["preds"])
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        ax=axes[idx],
        annot_kws={"size": 11, "weight": "bold"}
    )
    axes[idx].set_title(f"{name} (Val Set)", fontsize=11, fontweight="bold")
    axes[idx].set_xlabel("Predicted Label (0: Retained, 1: Churned)")
    axes[idx].set_ylabel("True Label (0: Retained, 1: Churned)")
    axes[idx].set_xticklabels(["Retained", "Churned"])
    axes[idx].set_yticklabels(["Retained", "Churned"])

plt.tight_layout()
chart_cm_path = os.path.join(CHARTS_DIR, "04_confusion_matrices.png")
plt.savefig(chart_cm_path, dpi=300)
plt.close()
print(f"      [SAVED] Confusion matrices chart: {chart_cm_path}")

# Figure 05: Feature Importances (from Random Forest)
rf_model = trained_models["Random Forest"]
importances = rf_model.feature_importances_
df_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(df_importance["Feature"], df_importance["Importance"], color="#3498db", edgecolor="#2980b9")
ax.set_title("Random Forest - Top Feature Importances (Gini Impurity)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Relative Importance Score")
ax.set_ylabel("Engineered & Transformed Features")
plt.tight_layout()
chart_fi_path = os.path.join(CHARTS_DIR, "05_feature_importance.png")
plt.savefig(chart_fi_path, dpi=300)
plt.close()
print(f"      [SAVED] Feature importance chart: {chart_fi_path}")

# ---------------------------------------------------------
# [6/6] Select Champion Model & Persist Artifacts
# ---------------------------------------------------------
print("\n[6/6] Selecting Champion Model and persisting artifacts for deployment...")

# Champion selection based on Validation F1 Score (balances false alarms and missed churn)
best_row = df_results.sort_values(by="Val F1", ascending=False).iloc[0]
champion_name = best_row["Model"]
champion_model = trained_models[champion_name]

print(f"\n      [CHAMPION MODEL]: '{champion_name}'")
print(f"         - Validation F1-Score : {best_row['Val F1']:.4f}")
print(f"         - Validation ROC-AUC  : {best_row['Val ROC-AUC']:.4f}")
print(f"         - Validation Accuracy : {best_row['Val Accuracy']:.4f}")

# 1. Save Champion Model
best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
joblib.dump(champion_model, best_model_path)
print(f"      [SAVED] Champion model binary: {best_model_path}")

# 2. Save All Candidate Models for flexibility in the Streamlit app
candidate_models_path = os.path.join(MODELS_DIR, "candidate_models.joblib")
joblib.dump(trained_models, candidate_models_path)
print(f"      [SAVED] All candidate models: {candidate_models_path}")

# 3. Save Benchmark Comparison JSON
benchmark_summary = {
    "champion_model": champion_name,
    "feature_names": feature_names,
    "trained_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "models": {}
}

for name, details in eval_details.items():
    benchmark_summary["models"][name] = {
        "train_time_seconds": details["train_time_sec"],
        "validation_metrics": {k: round(v, 4) for k, v in details["val"]["metrics"].items()},
        "external_test_metrics": {k: round(v, 4) for k, v in details["ext"]["metrics"].items()}
    }

metrics_json_path = os.path.join(MODELS_DIR, "model_comparison.json")
with open(metrics_json_path, "w") as f:
    json.dump(benchmark_summary, f, indent=4)
print(f"      [SAVED] Benchmark metrics summary: {metrics_json_path}")

print("\n" + "=" * 75)
print("  STEP 4 COMPLETE! ALL MODELS BENCHMARKED & ARTIFACTS PERSISTED.")
print("  Next Step: Ready to build interactive Streamlit web app (05_app.py)!")
print("=" * 75)
