"""
app.py
------
Interactive Streamlit Web Application for Customer Churn Prediction.

Phase 8 Implementation:
- Loads fitted Scikit-Learn preprocessor ('models/preprocessor.joblib')
- Loads champion trained model ('models/best_model.joblib')
- Dynamic feature engineering: Spend_per_Tenure, Support_Intensity, High_Delay_Flag
- Modern, responsive SaaS-style UI with real-time risk assessment & prescriptive retention actions
- Comprehensive tabs: Real-time Predictor, Model Benchmark Insights, Batch CSV Scoring & EDA
"""

import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="ChurnGuard | AI Customer Churn Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich, polished aesthetics
CUSTOM_CSS = """
<style>
    /* Global Font & Theme Tuning */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* Glassmorphic Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }

    /* Status Badges */
    .status-badge-low {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: #ffffff;
        padding: 8px 16px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.05rem;
        display: inline-block;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35);
    }
    .status-badge-med {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        color: #ffffff;
        padding: 8px 16px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.05rem;
        display: inline-block;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 14px rgba(245, 158, 11, 0.35);
    }
    .status-badge-high {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: #ffffff;
        padding: 8px 16px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.05rem;
        display: inline-block;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.35);
    }

    /* Feature Pill */
    .feature-pill {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 8px;
        padding: 6px 12px;
        margin: 4px;
        font-size: 0.85rem;
        font-weight: 500;
        color: #c7d2fe;
    }

    /* Action Item Box */
    .action-box {
        border-left: 4px solid #6366f1;
        background: rgba(99, 102, 241, 0.08);
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-top: 10px;
        font-size: 0.92rem;
    }

    /* Subtitle text */
    .subtext {
        color: #94a3b8;
        font-size: 0.9rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Resource Caching & Model Loaders
# ---------------------------------------------------------
MODELS_DIR = "models"
CHARTS_DIR = "charts"
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
ALL_MODELS_PATH = os.path.join(MODELS_DIR, "candidate_models.joblib")
COMPARISON_PATH = os.path.join(MODELS_DIR, "model_comparison.json")

@st.cache_resource(show_spinner="Loading machine learning models & preprocessor...")
def load_ml_assets():
    """Load preprocessor pipeline and trained models."""
    if not os.path.exists(PREPROCESSOR_PATH):
        st.error(f"Preprocessor not found at '{PREPROCESSOR_PATH}'. Please run scripts/03_preprocessing.py first.")
        st.stop()
    if not os.path.exists(BEST_MODEL_PATH):
        st.error(f"Best model not found at '{BEST_MODEL_PATH}'. Please run scripts/04_train_models.py first.")
        st.stop()

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    best_model = joblib.load(BEST_MODEL_PATH)

    candidate_models = {}
    if os.path.exists(ALL_MODELS_PATH):
        try:
            candidate_models = joblib.load(ALL_MODELS_PATH)
        except Exception:
            candidate_models = {"Gradient Boosting (Champion)": best_model}

    benchmark_data = {}
    if os.path.exists(COMPARISON_PATH):
        try:
            with open(COMPARISON_PATH, "r") as f:
                benchmark_data = json.load(f)
        except Exception:
            pass

    return preprocessor, best_model, candidate_models, benchmark_data

preprocessor, best_model, candidate_models, benchmark_data = load_ml_assets()

# ---------------------------------------------------------
# Helper Functions: Feature Engineering & Inference
# ---------------------------------------------------------
def compute_engineered_features(raw_data: dict) -> pd.DataFrame:
    """
    Computes the 3 domain-engineered features:
    1. Spend_per_Tenure = Total Spend / (Tenure + 1)
    2. Support_Intensity = Support Calls / (Usage Frequency + 1)
    3. High_Delay_Flag = 1 if Payment Delay > 15 else 0
    """
    df = pd.DataFrame([raw_data])
    df["Spend_per_Tenure"] = df["Total Spend"] / (df["Tenure"] + 1)
    df["Support_Intensity"] = df["Support Calls"] / (df["Usage Frequency"] + 1)
    df["High_Delay_Flag"] = (df["Payment Delay"] > 15).astype(int)
    return df

def predict_churn(customer_df: pd.DataFrame, model):
    """Transform customer dataframe and predict churn probability."""
    X_processed = preprocessor.transform(customer_df)
    churn_prob = model.predict_proba(X_processed)[0, 1]
    churn_pred = int(model.predict(X_processed)[0])
    return churn_pred, float(churn_prob)

# ---------------------------------------------------------
# Sidebar: Customer Profile Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=54)
    st.title("Customer Profile")
    st.caption("Adjust customer attributes to test churn prediction.")

    # Persona Quick Loader
    st.markdown("### ⚡ Quick Persona Presets")
    persona_options = [
        "Custom Input",
        "Low-Risk Loyal Customer",
        "Moderate-Risk Standard Customer",
        "Critical High-Risk Customer"
    ]
    selected_persona = st.selectbox(
        "Load Preset Persona:",
        persona_options,
        index=0,
        help="Instantly populate attributes based on realistic customer profiles."
    )

    # Defaults mapping
    presets = {
        "Low-Risk Loyal Customer": {
            "gender": "Female", "age": 42, "tenure": 45,
            "usage": 24, "calls": 1, "delay": 2, "spend": 850.0,
            "interaction": 5, "subscription": "Premium", "contract": "Annual"
        },
        "Moderate-Risk Standard Customer": {
            "gender": "Male", "age": 36, "tenure": 18,
            "usage": 14, "calls": 3, "delay": 8, "spend": 550.0,
            "interaction": 12, "subscription": "Standard", "contract": "Quarterly"
        },
        "Critical High-Risk Customer": {
            "gender": "Male", "age": 52, "tenure": 4,
            "usage": 6, "calls": 7, "delay": 22, "spend": 280.0,
            "interaction": 26, "subscription": "Basic", "contract": "Monthly"
        }
    }

    vals = presets.get(selected_persona, {
        "gender": "Female", "age": 38, "tenure": 24,
        "usage": 16, "calls": 2, "delay": 5, "spend": 650.0,
        "interaction": 10, "subscription": "Standard", "contract": "Annual"
    })

    st.markdown("---")
    st.markdown("### 👤 Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"], index=0 if vals["gender"] == "Female" else 1)
    age = st.slider("Age (Years)", min_value=18, max_value=65, value=int(vals["age"]), step=1)

    st.markdown("### 📋 Subscription & Contract")
    contract_opts = ["Annual", "Quarterly", "Monthly"]
    contract_idx = contract_opts.index(vals["contract"]) if vals["contract"] in contract_opts else 0
    contract_length = st.selectbox("Contract Length", contract_opts, index=contract_idx)

    sub_opts = ["Basic", "Standard", "Premium"]
    sub_idx = sub_opts.index(vals["subscription"]) if vals["subscription"] in sub_opts else 1
    subscription_type = st.selectbox("Subscription Tier", sub_opts, index=sub_idx)

    tenure = st.slider("Tenure (Months with Company)", min_value=1, max_value=60, value=int(vals["tenure"]), step=1)

    st.markdown("### 📈 Usage & Engagement")
    usage_freq = st.slider("Usage Frequency (Sessions/Month)", min_value=1, max_value=30, value=int(vals["usage"]), step=1)
    last_interaction = st.slider("Last Interaction (Days Ago)", min_value=1, max_value=30, value=int(vals["interaction"]), step=1)

    st.markdown("### 💳 Billing & Customer Support")
    support_calls = st.slider("Support Calls Logged", min_value=0, max_value=10, value=int(vals["calls"]), step=1)
    payment_delay = st.slider("Payment Delay (Days)", min_value=0, max_value=30, value=int(vals["delay"]), step=1)
    total_spend = st.number_input("Total Spend ($)", min_value=100.0, max_value=1000.0, value=float(vals["spend"]), step=25.0)

    st.markdown("---")
    st.markdown("### ⚙️ Inference Model")
    available_model_names = list(candidate_models.keys()) if candidate_models else ["Champion: Gradient Boosting"]
    selected_model_name = st.selectbox(
        "Active Classifier:",
        available_model_names,
        index=0,
        help="Select which trained model runs the prediction."
    )
    active_model = candidate_models.get(selected_model_name, best_model)

# Construct raw customer dictionary
customer_raw = {
    "Age": age,
    "Tenure": tenure,
    "Usage Frequency": usage_freq,
    "Support Calls": support_calls,
    "Payment Delay": payment_delay,
    "Total Spend": total_spend,
    "Last Interaction": last_interaction,
    "Gender": gender,
    "Subscription Type": subscription_type,
    "Contract Length": contract_length
}

# Dynamically compute engineered features
customer_df = compute_engineered_features(customer_raw)

# Run Inference
churn_pred, churn_prob = predict_churn(customer_df, active_model)
retention_prob = 1.0 - churn_prob

# ---------------------------------------------------------
# Main Page Header & Hero Section
# ---------------------------------------------------------
col_header_a, col_header_b = st.columns([3, 1])
with col_header_a:
    st.title("🛡️ Customer Churn Intelligence System")
    st.markdown(
        "Real-time AI-powered customer attrition risk forecasting, automated feature engineering, "
        "and prescriptive retention intelligence."
    )
with col_header_b:
    st.caption("Active Champion Algorithm")
    st.info(f"🚀 **{selected_model_name}**")

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab_prediction, tab_benchmark, tab_batch, tab_eda = st.tabs([
    "🔮 Live Churn Prediction",
    "📊 Model Performance & Benchmarks",
    "📁 Batch Scoring (CSV)",
    "📑 Exploratory Insights (EDA)"
])

# =========================================================
# TAB 1: Live Churn Prediction
# =========================================================
with tab_prediction:
    st.markdown("## Customer Risk Assessment")

    # Risk level classification
    if churn_prob < 0.35:
        risk_level = "LOW RISK"
        badge_class = "status-badge-low"
        risk_icon = "🟢"
        risk_color = "#10b981"
        status_headline = "Low Churn Propensity: Customer is Well-Retained"
    elif churn_prob < 0.65:
        risk_level = "MODERATE RISK"
        badge_class = "status-badge-med"
        risk_icon = "🟡"
        risk_color = "#f59e0b"
        status_headline = "Moderate Churn Propensity: Action Recommended"
    else:
        risk_level = "HIGH RISK"
        badge_class = "status-badge-high"
        risk_icon = "🔴"
        risk_color = "#ef4444"
        status_headline = "Critical Churn Alert: Immediate Intervention Required"

    # Top Results Banner
    col_banner_1, col_banner_2, col_banner_3 = st.columns([1.5, 1.2, 1.2])

    with col_banner_1:
        st.markdown(f'<div class="{badge_class}">{risk_icon} {risk_level}</div>', unsafe_allow_html=True)
        st.markdown(f"### {status_headline}")
        st.caption(f"Evaluated using {selected_model_name} based on 10 behavioral & 3 engineered features.")

    with col_banner_2:
        st.metric(
            label="Churn Probability",
            value=f"{churn_prob * 100:.2f}%",
            delta=f"{'+' if churn_prob > 0.5 else '-'}{abs(churn_prob - 0.5)*100:.1f}% from baseline",
            delta_color="inverse"
        )

    with col_banner_3:
        st.metric(
            label="Retention Probability",
            value=f"{retention_prob * 100:.2f}%",
            delta=f"{'+' if retention_prob > 0.5 else '-'}{abs(retention_prob - 0.5)*100:.1f}% from baseline",
            delta_color="normal"
        )

    # Progress bar visualizer
    st.markdown(f"**Churn Risk Spectrum:**")
    st.progress(float(np.clip(churn_prob, 0.0, 1.0)))

    st.markdown("---")

    # Dynamic Feature Engineering Real-time Showcase
    st.markdown("### ⚙️ Real-Time Dynamic Feature Engineering")
    st.markdown(
        "These 3 domain features were dynamically derived from your sidebar inputs in real time "
        "and passed directly into the preprocessor pipeline:"
    )

    col_feat1, col_feat2, col_feat3 = st.columns(3)

    with col_feat1:
        st.markdown(
            f"""
            <div class="metric-card">
                <span class="subtext">Feature 1: Value Velocity</span>
                <h4 style="margin: 6px 0;">Spend per Month of Tenure</h4>
                <h2 style="color: #6366f1; margin: 4px 0;">${customer_df['Spend_per_Tenure'].iloc[0]:.2f} /mo</h2>
                <span class="feature-pill">Formula: Total Spend / (Tenure + 1)</span>
                <p style="font-size: 0.82rem; color: #94a3b8; margin-top: 8px;">
                    Measures capital commitment rate over the customer's lifespan.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_feat2:
        st.markdown(
            f"""
            <div class="metric-card">
                <span class="subtext">Feature 2: Friction Density</span>
                <h4 style="margin: 6px 0;">Support Call Intensity</h4>
                <h2 style="color: #6366f1; margin: 4px 0;">{customer_df['Support_Intensity'].iloc[0]:.2f}</h2>
                <span class="feature-pill">Formula: Support Calls / (Usage + 1)</span>
                <p style="font-size: 0.82rem; color: #94a3b8; margin-top: 8px;">
                    Ratio of support issues relative to active usage frequency.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_feat3:
        high_delay_val = int(customer_df['High_Delay_Flag'].iloc[0])
        flag_color = "#ef4444" if high_delay_val == 1 else "#10b981"
        flag_text = "RISK DETECTED (> 15 Days)" if high_delay_val == 1 else "NORMAL (<= 15 Days)"
        st.markdown(
            f"""
            <div class="metric-card">
                <span class="subtext">Feature 3: Billing Distress Flag</span>
                <h4 style="margin: 6px 0;">Payment Delay Risk Indicator</h4>
                <h2 style="color: {flag_color}; margin: 4px 0;">{flag_text}</h2>
                <span class="feature-pill">Formula: 1 if Delay > 15 else 0</span>
                <p style="font-size: 0.82rem; color: #94a3b8; margin-top: 8px;">
                    Binary trigger identifying high-risk billing delinquency.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Two-Column Layout: Customer Profile Card vs Prescriptive Recommendations
    col_summary, col_action = st.columns([1, 1])

    with col_summary:
        st.markdown("### 📋 Input Profile Summary")
        summary_table = pd.DataFrame({
            "Attribute": [
                "Demographics", "Contract Length", "Subscription Tier", "Tenure",
                "Total Spend", "Usage Frequency", "Support Calls", "Payment Delay", "Last Interaction"
            ],
            "Value": [
                f"{gender}, {age} yrs old",
                f"{contract_length}",
                f"{subscription_type}",
                f"{tenure} months",
                f"${total_spend:,.2f}",
                f"{usage_freq} sessions/month",
                f"{support_calls} calls logged",
                f"{payment_delay} days overdue",
                f"{last_interaction} days ago"
            ]
        })
        st.dataframe(summary_table, hide_index=True, use_container_width=True)

    with col_action:
        st.markdown("### 🎯 Prescriptive Retention Playbook")
        st.markdown("Data-driven business countermeasures tailored to this customer's profile:")

        if churn_prob > 0.50:
            if payment_delay > 15:
                st.markdown(
                    """
                    <div class="action-box">
                        <strong>💳 Overdue Payment Intervention:</strong> Payment delay exceeds 15 days.
                        Initiate proactive payment restructuring, waive late penalties, and offer automated ACH billing discounts.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            if support_calls >= 4:
                st.markdown(
                    """
                    <div class="action-box">
                        <strong>🎧 High Support Friction:</strong> Over 4 support calls registered.
                        Assign a Dedicated Customer Success Manager to review outstanding support tickets within 24 hours.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            if contract_length == "Monthly":
                st.markdown(
                    """
                    <div class="action-box">
                        <strong>📄 Contract Upgrade Incentive:</strong> Monthly subscribers churn at a 100% rate in historical data.
                        Offer an Annual Contract upgrade incentive with a 20% loyalty discount.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            if last_interaction > 20:
                st.markdown(
                    """
                    <div class="action-box">
                        <strong>📢 Re-engagement Outreach:</strong> Last interaction was over 20 days ago.
                        Trigger automated re-onboarding email sequence with personalized feature spotlights.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                """
                <div class="action-box" style="border-left-color: #10b981; background: rgba(16, 185, 129, 0.08);">
                    <strong>🌟 VIP Loyalty Engagement:</strong> Customer exhibits strong retention signals.
                    Enroll in exclusive Beta features, invite to annual product advisory council, or offer referral bonus rewards.
                </div>
                """,
                unsafe_allow_html=True
            )
            if subscription_type != "Premium":
                st.markdown(
                    """
                    <div class="action-box" style="border-left-color: #6366f1;">
                        <strong>🚀 Upsell Opportunity:</strong> High satisfaction and low friction indicate readiness for
                        an upgrade to the Premium Tier.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# =========================================================
# TAB 2: Model Benchmarks & Explanations
# =========================================================
with tab_benchmark:
    st.markdown("## 📊 Model Training & Evaluation Benchmarks")
    st.markdown(
        "Comprehensive performance metrics across 4 machine learning algorithms trained on **440,832** customer records "
        "and validated on a stratified holdout split."
    )

    if benchmark_data and "models" in benchmark_data:
        benchmark_rows = []
        for model_name, info in benchmark_data["models"].items():
            val = info["validation_metrics"]
            benchmark_rows.append({
                "Algorithm": model_name,
                "Train Time (s)": info["train_time_seconds"],
                "Val Accuracy": f"{val['accuracy']*100:.2f}%",
                "Val Precision": f"{val['precision']*100:.2f}%",
                "Val Recall": f"{val['recall']*100:.2f}%",
                "Val F1-Score": f"{val['f1']:.4f}",
                "Val ROC-AUC": f"{val['roc_auc']:.4f}"
            })
        df_bench = pd.DataFrame(benchmark_rows)
        st.dataframe(df_bench, hide_index=True, use_container_width=True)
    else:
        st.info("Run `scripts/04_train_models.py` to populate real-time benchmark summary.")

    st.markdown("---")
    st.markdown("### 📈 Visual Evaluation Artifacts")

    col_chart_1, col_chart_2 = st.columns(2)

    chart_metrics = os.path.join(CHARTS_DIR, "03_model_metrics_comparison.png")
    chart_cm = os.path.join(CHARTS_DIR, "04_confusion_matrices.png")
    chart_fi = os.path.join(CHARTS_DIR, "05_feature_importance.png")

    with col_chart_1:
        if os.path.exists(chart_metrics):
            st.image(chart_metrics, caption="Figure 3: Multi-Metric Benchmark Comparison across Algorithms")
        else:
            st.warning("Chart 03_model_metrics_comparison.png not found.")

    with col_chart_2:
        if os.path.exists(chart_cm):
            st.image(chart_cm, caption="Figure 4: Confusion Matrices for All 4 Candidate Classifiers")
        else:
            st.warning("Chart 04_confusion_matrices.png not found.")

    st.markdown("---")
    st.markdown("### 🔍 Model Explainability & Key Drivers")

    col_fi_img, col_fi_desc = st.columns([1.2, 1])

    with col_fi_img:
        if os.path.exists(chart_fi):
            st.image(chart_fi, caption="Figure 5: Feature Importances (Gini Impurity from Random Forest)")
        else:
            st.warning("Chart 05_feature_importance.png not found.")

    with col_fi_desc:
        st.markdown(
            """
            #### Key Takeaways for Data Science Stakeholders:
            1. **Contract Length (Monthly vs Annual):** Monthly contract customers exhibit the highest churn vulnerability.
            2. **Support Calls:** Customers exceeding 4-5 customer service calls cross a sharp friction threshold.
            3. **Payment Delay:** Delays greater than 15 days serve as an early indicator of imminent subscription cancellation.
            4. **Engineered Features:** Combining `Spend_per_Tenure` and `Support_Intensity` significantly strengthens linear and tree-based decision boundaries.
            """
        )

# =========================================================
# TAB 3: Batch CSV Scoring
# =========================================================
with tab_batch:
    st.markdown("## 📁 Batch Customer Scoring")
    st.markdown(
        "Upload a CSV file containing multiple customer records to perform automated bulk feature engineering "
        "and churn risk scoring."
    )

    # Sample CSV Download Template
    sample_csv_data = pd.DataFrame([
        {
            "CustomerID": 101, "Age": 32, "Gender": "Female", "Tenure": 12,
            "Usage Frequency": 15, "Support Calls": 1, "Payment Delay": 3,
            "Subscription Type": "Standard", "Contract Length": "Annual",
            "Total Spend": 550.0, "Last Interaction": 8
        },
        {
            "CustomerID": 102, "Age": 55, "Gender": "Male", "Tenure": 3,
            "Usage Frequency": 5, "Support Calls": 7, "Payment Delay": 24,
            "Subscription Type": "Basic", "Contract Length": "Monthly",
            "Total Spend": 220.0, "Last Interaction": 26
        },
        {
            "CustomerID": 103, "Age": 41, "Gender": "Female", "Tenure": 36,
            "Usage Frequency": 22, "Support Calls": 3, "Payment Delay": 10,
            "Subscription Type": "Premium", "Contract Length": "Quarterly",
            "Total Spend": 780.0, "Last Interaction": 14
        }
    ])

    csv_buffer = io.StringIO()
    sample_csv_data.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=csv_buffer.getvalue(),
        file_name="sample_customers_to_score.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload customer CSV file:", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(batch_df):,} customer records successfully.")

            required_cols = [
                "Age", "Gender", "Tenure", "Usage Frequency",
                "Support Calls", "Payment Delay", "Subscription Type",
                "Contract Length", "Total Spend", "Last Interaction"
            ]
            missing_cols = [c for c in required_cols if c not in batch_df.columns]

            if missing_cols:
                st.error(f"Missing required columns in uploaded CSV: {missing_cols}")
            else:
                # Dynamic Feature Engineering for batch
                scored_df = batch_df.copy()
                scored_df["Spend_per_Tenure"] = scored_df["Total Spend"] / (scored_df["Tenure"] + 1)
                scored_df["Support_Intensity"] = scored_df["Support Calls"] / (scored_df["Usage Frequency"] + 1)
                scored_df["High_Delay_Flag"] = (scored_df["Payment Delay"] > 15).astype(int)

                # Inference
                X_batch_proc = preprocessor.transform(scored_df)
                batch_probs = active_model.predict_proba(X_batch_proc)[:, 1]
                batch_preds = active_model.predict(X_batch_proc)

                scored_df["Churn_Probability"] = np.round(batch_probs, 4)
                scored_df["Predicted_Churn"] = batch_preds
                scored_df["Risk_Category"] = pd.cut(
                    batch_probs,
                    bins=[-0.01, 0.35, 0.65, 1.0],
                    labels=["Low Risk", "Moderate Risk", "High Risk"]
                )

                st.markdown("### 📊 Batch Scoring Results Summary")
                col_b1, col_b2, col_b3 = st.columns(3)
                high_risk_count = int((scored_df["Risk_Category"] == "High Risk").sum())
                med_risk_count = int((scored_df["Risk_Category"] == "Moderate Risk").sum())
                low_risk_count = int((scored_df["Risk_Category"] == "Low Risk").sum())

                with col_b1:
                    st.metric("High Risk Customers", f"{high_risk_count:,}", f"{high_risk_count/len(scored_df)*100:.1f}%")
                with col_b2:
                    st.metric("Moderate Risk Customers", f"{med_risk_count:,}", f"{med_risk_count/len(scored_df)*100:.1f}%")
                with col_b3:
                    st.metric("Low Risk Customers", f"{low_risk_count:,}", f"{low_risk_count/len(scored_df)*100:.1f}%")

                st.dataframe(
                    scored_df[[
                        c for c in ["CustomerID", "Age", "Gender", "Contract Length",
                                   "Total Spend", "Churn_Probability", "Risk_Category"] if c in scored_df.columns
                    ]],
                    use_container_width=True
                )

                # Export scored CSV
                out_buffer = io.StringIO()
                scored_df.to_csv(out_buffer, index=False)
                st.download_button(
                    label="💾 Export Scored Customer List (CSV)",
                    data=out_buffer.getvalue(),
                    file_name="churn_predictions_export.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")

# =========================================================
# TAB 4: Exploratory Data Analysis (EDA) Insights
# =========================================================
with tab_eda:
    st.markdown("## 📑 Exploratory Data Analysis Insights")
    st.markdown(
        "Key statistical discoveries and visual distributions from the training set exploration (`scripts/02_data_analysis.py`)."
    )

    col_eda_1, col_eda_2 = st.columns(2)

    chart_eda_1 = os.path.join(CHARTS_DIR, "01_support_and_payment_delay.png")
    chart_eda_2 = os.path.join(CHARTS_DIR, "02_contract_and_subscription_churn.png")

    with col_eda_1:
        if os.path.exists(chart_eda_1):
            st.image(chart_eda_1, caption="Figure 1: Behavioral Differences in Support Calls & Payment Delay")
        else:
            st.info("Chart 01_support_and_payment_delay.png not found.")

    with col_eda_2:
        if os.path.exists(chart_eda_2):
            st.image(chart_eda_2, caption="Figure 2: Churn Distribution by Contract Length and Subscription Type")
        else:
            st.info("Chart 02_contract_and_subscription_churn.png not found.")

    st.markdown("---")
    st.markdown(
        """
        ### 💡 Strategic Business Takeaways:
        - **Monthly Contracts vs Annual Contracts:** Customers on monthly contracts experience virtually guaranteed churn without proactive engagement, whereas annual contracts yield high retention stability.
        - **Support Friction Points:** Retained customers average ~1.8 support calls, whereas churned customers average ~4.9 calls. Early alerts should trigger when a customer logs their 3rd call.
        - **Payment Delays:** Customers who incur payment delays beyond 15 days have an 85%+ churn conversion rate, highlighting payment process friction as a leading indicator.
        """
    )
