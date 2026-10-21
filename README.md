# 🛡️ Customer Churn Intelligence & Risk Forecasting System

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66.0-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-3.0.6-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

An end-to-end, enterprise-grade machine learning system designed to predict customer churn, identify critical behavioral attrition triggers, and deliver real-time prescriptive retention countermeasures. Built on a dataset of **500,000+** customer records, the system incorporates data integrity validation, leakage-free preprocessing, multi-model benchmarking, and an interactive **Streamlit** dashboard.

---

## 📌 Table of Contents

- [Business Context & Problem Statement](#-business-context--problem-statement)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Exploratory Data Analysis (EDA) Insights](#-exploratory-data-analysis-eda-insights)
- [Feature Engineering Strategy](#-feature-engineering-strategy)
- [Model Training & Benchmark Comparison](#-model-training--benchmark-comparison)
- [Streamlit Web Application](#-streamlit-web-application)
- [Installation & Quickstart Guide](#-installation--quickstart-guide)
- [Key Data Science Takeaways](#-key-data-science-takeaways)

---

## 💼 Business Context & Problem Statement

Acquiring new customers costs 5x to 7x more than retaining existing ones. In SaaS, telecommunications, and subscription services, customer churn directly degrades recurring revenue and customer lifetime value (LTV).

### Core Objectives:
1. **Accurate Forecasting**: Classify customer churn propensity with high precision and recall to avoid both missed churners (false negatives) and wasted retention spend (false positives).
2. **Early Friction Detection**: Automatically identify friction points such as payment delays, escalating customer support calls, and contract mismatches.
3. **Prescriptive Action**: Automatically trigger tailored retention playbooks (contract incentives, dedicated CSM outreach, payment plan restructuring) based on individual customer profiles.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        A[Raw Customer Records<br/>440k+ Train / 64k+ Test] --> B[Data Cleaning & Null Pruning<br/>scripts/02_data_analysis.py]
    end

    subgraph Feature Engineering & Pipeline
        B --> C[Domain Feature Engineering<br/>Spend/Tenure, Support Intensity, Delay Flag]
        C --> D[Leakage-Free ColumnTransformer<br/>StandardScaler + OneHotEncoder]
        D --> E[Processed Artifacts<br/>models/preprocessor.joblib<br/>data/processed/churn_processed_data.npz]
    end

    subgraph Machine Learning Benchmarking
        E --> F[Stratified Train/Val Split<br/>80% Train | 20% Validation]
        F --> G1[Logistic Regression<br/>Linear Baseline]
        F --> G2[Decision Tree<br/>White-box Rules]
        F --> G3[Random Forest<br/>100-Tree Bagging]
        F --> G4[Gradient Boosting<br/>HistGradientBoosting]
        G1 & G2 & G3 & G4 --> H[Multi-Metric Evaluation<br/>F1, ROC-AUC, Precision, Recall]
        H --> I[Champion Selection<br/>models/best_model.joblib]
        H --> J[Diagnostic Plots<br/>charts/03, 04, 05]
    end

    subgraph Production Deployment
        I & D --> K[Interactive Streamlit App<br/>app.py]
        K --> L1[Real-Time Single Profile Prediction]
        K --> L2[Dynamic Feature Computation]
        K --> L3[Prescriptive Retention Playbook]
        K --> L4[Bulk CSV Scoring & Export]
    end
```

---

## 📂 Repository Structure

```text
Customer Churn Prediction System/
│
├── archive/                               # Raw Kaggle customer datasets (git-ignored)
│   ├── customer_churn_dataset-training-master.csv   # 440,833 records
│   └── customer_churn_dataset-testing-master.csv    # 64,374 records
│
├── charts/                                # Publication-ready visualization artifacts
│   ├── 01_support_and_payment_delay.png   # EDA: Support calls & delay vs churn
│   ├── 02_contract_and_subscription_churn.png # EDA: Contract & tier distribution
│   ├── 03_model_metrics_comparison.png    # ML: Metric benchmark comparison
│   ├── 04_confusion_matrices.png          # ML: 2x2 confusion matrix grid
│   └── 05_feature_importance.png          # ML: Gini impurity feature ranking
│
├── data/
│   └── processed/                         # Processed feature arrays (git-ignored)
│       └── churn_processed_data.npz       # Compressed preprocessed matrices
│
├── models/                                # Serialized production artifacts
│   ├── best_model.joblib                  # Champion model binary (Gradient Boosting)
│   ├── candidate_models.joblib            # All trained candidate classifiers
│   ├── preprocessor.joblib                # Fitted Scikit-Learn ColumnTransformer
│   └── model_comparison.json              # Benchmark evaluation summary
│
├── scripts/                               # Core execution pipeline
│   ├── 02_data_analysis.py                # EDA and visual distribution profiling
│   ├── 03_preprocessing.py                # Cleaning, feature engineering & pipeline
│   └── 04_train_models.py                 # Multi-model training, benchmarking & persistence
│
├── app.py                                 # Interactive Streamlit web application
├── requirements.txt                       # Exact pinned environment dependencies
├── .gitignore                             # Production ignore rules
└── README.md                              # Comprehensive project documentation
```

---

## 📊 Exploratory Data Analysis (EDA) Insights

From the analysis conducted in `scripts/02_data_analysis.py`:

| Behavioral Feature | Retained Customers (Mean) | Churned Customers (Mean) | Observed Impact |
| :--- | :---: | :---: | :--- |
| **Support Calls** | 1.8 calls | 4.9 calls | **+172% increase** in support friction among churners |
| **Payment Delay** | 7.4 days | 17.2 days | **+132% increase**; delays > 15 days strongly correlate with churn |
| **Total Spend** | $742.80 | $547.20 | Churned customers spend **26.3% less** over their lifespan |
| **Contract Length** | Annual (0% churn) | Monthly (100% churn) | Monthly contract subscribers exhibit maximum churn vulnerability |

### Key Visual Artifacts:
- **`charts/01_support_and_payment_delay.png`**: Boxplot comparisons showing the shift in support call volume and payment delays for churned vs. retained users.
- **`charts/02_contract_and_subscription_churn.png`**: Categorical breakdown of churn rates across Monthly, Quarterly, and Annual contracts.

---

## ⚙️ Feature Engineering Strategy

To provide actionable signals for both linear and tree-based estimators, 3 domain-engineered features were constructed in `scripts/03_preprocessing.py`:

| Feature Name | Formula | Business Rationale |
| :--- | :--- | :--- |
| `Spend_per_Tenure` | $\frac{\text{Total Spend}}{\text{Tenure} + 1}$ | Captures customer value velocity (monthly capital commitment rate). |
| `Support_Intensity` | $\frac{\text{Support Calls}}{\text{Usage Frequency} + 1}$ | Measures friction relative to active engagement. |
| `High_Delay_Flag` | $\mathbb{I}(\text{Payment Delay} > 15)$ | Binary indicator identifying customers entering high billing delinquency risk. |

### Data Leakage Prevention:
All scalers (`StandardScaler`) and encoders (`OneHotEncoder`) are fit strictly on the **training set** inside a Scikit-Learn `ColumnTransformer`. Test sets and new user inputs are exclusively transformed using the pre-fitted parameters.

---

## 🏆 Model Training & Benchmark Comparison

Four models were trained on **352,665** training samples and validated on **88,167** stratified validation samples (`scripts/04_train_models.py`):

| Algorithm | Train Time | Val Accuracy | Val Precision | Val Recall | Val F1-Score | Val ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Gradient Boosting (`HistGradientBoosting`)** | **1.42s** | **99.99%** | **100.00%** | **99.98%** | **0.9999** | **1.0000** |
| 🥈 **Decision Tree** | 1.46s | 99.73% | 99.99% | 99.53% | 0.9976 | 0.9994 |
| 🥉 **Random Forest** | 5.96s | 99.26% | 99.99% | 98.71% | 0.9935 | 0.9999 |
| **Logistic Regression (Baseline)** | **0.43s** | 89.34% | 92.32% | 88.57% | 0.9041 | 0.9590 |

> **Champion Selection**: `HistGradientBoostingClassifier` was selected as the champion model. It achieved near-perfect F1-score (0.9999) and ROC-AUC (1.0000) while training in just **1.42 seconds** on 350k+ records.

---

## 💻 Streamlit Web Application

The production application in [app.py](file:///d:/Learning%20Workspace/My%20Degree%20%28DSBA%29/AI%20&%20Machine%20Learning%20Projects/Customer%20Churn%20Prediction%20System/app.py) provides a modern interface featuring:

1. **🔮 Live Churn Prediction**:
   - Sidebar customer attribute controls with **Quick Persona Presets** (*Low-Risk Loyal*, *Moderate-Risk Standard*, *Critical High-Risk*).
   - Dynamic real-time calculation of all 3 engineered features.
   - Dual-metric probability gauge (Churn Propensity % vs Retention Probability %).
   - **Prescriptive Retention Playbook**: Generates targeted business countermeasures (e.g. Contract upgrade offers, dedicated CSM review, ACH billing incentives).
2. **📊 Model Performance & Benchmarks**:
   - Interactive metric comparison tables and embedded visual charts (`charts/03`, `charts/04`, `charts/05`).
3. **📁 Batch Customer Scoring (CSV)**:
   - Upload bulk CSV files, automatically execute preprocessing, and export scored customer records with risk categories.
4. **📑 Exploratory Insights (EDA)**:
   - Visual summary of initial exploratory findings and retention patterns.

---

## 🚀 Installation & Quickstart Guide

### 1. Prerequisites
- Python 3.10 to 3.13
- Git

### 2. Clone Repository & Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/your-username/customer-churn-prediction-system.git
cd customer-churn-prediction-system

# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Execute Pipeline Steps Sequentially
```bash
# Step 1: Run Exploratory Data Analysis & generate EDA charts
python scripts/02_data_analysis.py

# Step 2: Clean data, engineer features, and fit preprocessor
python scripts/03_preprocessing.py

# Step 3: Train and benchmark candidate models & select champion
python scripts/04_train_models.py
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🎓 Key Data Science Takeaways

1. **Preventing Target & Feature Leakage**:
   Standardization and encoding must be fit solely on the training partition. Using `ColumnTransformer` guarantees consistent feature alignment between training, batch scoring, and live inference.
2. **Metrics Beyond Accuracy**:
   In churn modeling, false negatives (unidentified churners) directly lead to lost revenue. Optimizing for **Recall** and **F1-Score** ensures at-risk customers are identified early.
3. **Explainability & Actionability**:
   A prediction without explanation is difficult to act on. Extracting feature importances allows the system to bridge the gap between machine learning predictions and actionable customer success interventions.

---

## 📄 License & Attribution
- Dataset source: [Kaggle Customer Churn Dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset/code)
- Developed for DSBA (Data Science & Business Analytics) Machine Learning Portfolio.
