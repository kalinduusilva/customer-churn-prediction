"""
app.py
------
Customer Churn Intelligence & Risk Forecasting System
Interactive Executive Decision Support Dashboard

Author: Kalindu Mihiranga Silva
Credentials: 2nd-Year BSc (Hons) in Data Science & Business Analytics
Institution: General Sir John Kotelawala Defence University (KDU), Sri Lanka
Aspiring AI & Machine Learning Engineer

Features:
- Live Churn Probability Forecasting with Interactive Plotly Gauge (0-30% Safe, 30-70% Moderate, 70-100% Critical)
- Top-Row KPI Cards for the 3 Domain-Engineered Features with Formula Tooltips
- Interactive Live 'What-If' Retention Simulation Studio with Real-Time Risk Reduction Deltas
- Prescriptive Business Countermeasures & Action Playbooks
- Multi-Model Benchmarking & Visual Model Explainability (Gradient Boosting Champion)
- Enterprise Batch Customer CSV Scoring & Scored Data Export
- Exploratory Data Analysis (EDA) Insights & Distribution Profiling
"""

import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="ChurnGuard | AI Customer Churn Intelligence System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling: Glassmorphism, Modern Typography & Polish
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Clean header & remove default clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1320px;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(120deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .hero-sub {
        color: #94a3b8;
        font-size: 0.98rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Top-Row KPI Highlight Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.015) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.45);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.12);
    }

    .kpi-title {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }

    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 2px 0 6px 0;
        letter-spacing: -0.5px;
    }

    .kpi-formula-pill {
        display: inline-block;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.28);
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.76rem;
        font-family: 'JetBrains Mono', monospace;
        color: #c7d2fe;
        margin-top: 4px;
    }

    /* Status Badges */
    .status-badge-low {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: #ffffff;
        padding: 8px 18px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        letter-spacing: 0.4px;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35);
    }
    .status-badge-med {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        color: #ffffff;
        padding: 8px 18px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        letter-spacing: 0.4px;
        box-shadow: 0 4px 14px rgba(245, 158, 11, 0.35);
    }
    .status-badge-high {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: #ffffff;
        padding: 8px 18px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        letter-spacing: 0.4px;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.35);
    }

    /* Action Playbook Cards */
    .playbook-card {
        border-left: 4px solid #6366f1;
        background: rgba(99, 102, 241, 0.07);
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding: 14px 18px;
        border-radius: 0 12px 12px 0;
        margin-bottom: 12px;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .playbook-card-success {
        border-left-color: #10b981;
        background: rgba(16, 185, 129, 0.07);
    }

    .playbook-card-warning {
        border-left-color: #f59e0b;
        background: rgba(245, 158, 11, 0.07);
    }

    .playbook-card-danger {
        border-left-color: #ef4444;
        background: rgba(239, 68, 68, 0.07);
    }

    /* What-If Simulation Box */
    .whatif-box {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px solid rgba(129, 140, 248, 0.25);
        border-radius: 16px;
        padding: 22px 24px;
        margin-top: 18px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }

    /* Author Sidebar Card */
    .sidebar-author {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 18px;
        text-align: left;
    }
    .author-name {
        color: #ffffff;
        font-weight: 700;
        font-size: 0.95rem;
        margin: 0;
    }
    .author-role {
        color: #94a3b8;
        font-size: 0.8rem;
        margin: 2px 0 8px 0;
        line-height: 1.3;
    }
    .author-tag {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34d399;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        padding: 26px 0 10px 0;
        color: #64748b;
        font-size: 0.85rem;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        margin-top: 40px;
    }
    .app-footer a {
        color: #818cf8;
        text-decoration: none;
    }
    .app-footer a:hover {
        text-decoration: underline;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Resource Caching & Asset Loaders
# ---------------------------------------------------------
MODELS_DIR = "models"
CHARTS_DIR = "charts"
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
ALL_MODELS_PATH = os.path.join(MODELS_DIR, "candidate_models.joblib")
COMPARISON_PATH = os.path.join(MODELS_DIR, "model_comparison.json")

@st.cache_resource(show_spinner="Loading machine learning models & preprocessor pipeline...")
def load_ml_assets():
    """Load preprocessor pipeline, champion model, candidates, and benchmark stats."""
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
    else:
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

def create_churn_gauge_chart(probability: float) -> go.Figure:
    """
    Constructs an interactive executive Plotly gauge chart with 3 colored risk zones:
    - 0% to 30%: Safe Retained Zone (Emerald Green)
    - 30% to 70%: Moderate Risk Zone (Amber)
    - 70% to 100%: Critical Churn Alert Zone (Crimson)
    """
    prob_pct = round(probability * 100, 2)

    if prob_pct < 30.0:
        bar_color = "#10b981"  # Emerald
    elif prob_pct < 70.0:
        bar_color = "#f59e0b"  # Amber
    else:
        bar_color = "#ef4444"  # Crimson

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "<b>ATTRITION PROPENSITY</b><br><span style='font-size:0.75em;color:#94a3b8'>Model Forecast Probability</span>",
            'font': {'size': 18, 'color': '#f8fafc', 'family': 'Plus Jakarta Sans'}
        },
        delta={
            'reference': 50.0,
            'increasing': {'color': '#ef4444'},
            'decreasing': {'color': '#10b981'},
            'suffix': '% vs 50% baseline'
        },
        number={'suffix': '%', 'font': {'size': 38, 'color': '#ffffff', 'family': 'Plus Jakarta Sans'}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1.5,
                'tickcolor': '#64748b',
                'tickvals': [0, 15, 30, 50, 70, 85, 100],
                'ticktext': ['0%', 'Safe', '30%', '50%', '70%', 'Alert', '100%']
            },
            'bar': {'color': bar_color, 'thickness': 0.32},
            'bgcolor': 'rgba(255, 255, 255, 0.04)',
            'borderwidth': 1,
            'bordercolor': 'rgba(255, 255, 255, 0.12)',
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.22)'},
                {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.22)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.25)'}
            ],
            'threshold': {
                'line': {'color': '#ef4444', 'width': 3},
                'thickness': 0.75,
                'value': 70.0
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e2e8f0', 'family': 'Plus Jakarta Sans'},
        height=270,
        margin=dict(l=25, r=25, t=55, b=25)
    )
    return fig

# ---------------------------------------------------------
# Sidebar: Profile Configuration & Developer Branding
# ---------------------------------------------------------
with st.sidebar:
    # Developer Branding Badge
    st.markdown(
        """
        <div class="sidebar-author">
            <span class="author-tag">PORTFOLIO PROJECT</span>
            <h4 class="author-name" style="margin-top: 6px;">Kalindu Mihiranga Silva</h4>
            <p class="author-role">
                2nd-Year BSc (Hons) Data Science & Business Analytics<br>
                General Sir John Kotelawala Defence University, Sri Lanka
            </p>
            <div style="font-size: 0.78rem;">
                <a href="https://linkedin.com/in/kalindusilva" target="_blank" style="color: #818cf8; text-decoration: none; margin-right: 12px;">🔗 LinkedIn</a>
                <a href="https://github.com/kalinduusilva" target="_blank" style="color: #818cf8; text-decoration: none;">🐙 GitHub</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### ⚡ Quick Persona Presets")
    persona_options = [
        "Custom Input",
        "Low-Risk Loyal Customer",
        "Moderate-Risk Standard Customer",
        "Critical High-Risk Customer"
    ]
    selected_persona = st.selectbox(
        "Load Preset Profile:",
        persona_options,
        index=0,
        help="Instantly test realistic customer profiles across the risk spectrum."
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
    st.markdown("### 👤 Demographics & Identity")
    gender = st.selectbox("Gender", ["Female", "Male"], index=0 if vals["gender"] == "Female" else 1)
    age = st.slider("Age (Years)", min_value=18, max_value=65, value=int(vals["age"]), step=1)

    st.markdown("### 📋 Contract & Subscription")
    contract_opts = ["Annual", "Quarterly", "Monthly"]
    contract_idx = contract_opts.index(vals["contract"]) if vals["contract"] in contract_opts else 0
    contract_length = st.selectbox("Contract Length", contract_opts, index=contract_idx)

    sub_opts = ["Basic", "Standard", "Premium"]
    sub_idx = sub_opts.index(vals["subscription"]) if vals["subscription"] in sub_opts else 1
    subscription_type = st.selectbox("Subscription Tier", sub_opts, index=sub_idx)

    tenure = st.slider("Tenure (Months with Company)", min_value=1, max_value=60, value=int(vals["tenure"]), step=1)

    st.markdown("### 📈 Usage & Behavioral Engagement")
    usage_freq = st.slider("Usage Frequency (Sessions/Month)", min_value=1, max_value=30, value=int(vals["usage"]), step=1)
    last_interaction = st.slider("Last Interaction (Days Ago)", min_value=1, max_value=30, value=int(vals["interaction"]), step=1)

    st.markdown("### 💳 Billing & Support Friction")
    support_calls = st.slider("Support Calls Logged", min_value=0, max_value=10, value=int(vals["calls"]), step=1)
    payment_delay = st.slider("Payment Delay (Days Overdue)", min_value=0, max_value=30, value=int(vals["delay"]), step=1)
    total_spend = st.number_input("Total Spend ($ USD)", min_value=100.0, max_value=1000.0, value=float(vals["spend"]), step=25.0)

    st.markdown("---")
    st.markdown("### ⚙️ Inference Classifier")
    available_model_names = list(candidate_models.keys()) if candidate_models else ["Gradient Boosting (Champion)"]
    selected_model_name = st.selectbox(
        "Active Model:",
        available_model_names,
        index=0,
        help="Select which machine learning estimator evaluates the profile."
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

# Dynamically compute engineered features for current customer
customer_df = compute_engineered_features(customer_raw)

# Run Inference
churn_pred, churn_prob = predict_churn(customer_df, active_model)
retention_prob = 1.0 - churn_prob

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-banner">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h1 class="hero-title">🛡️ ChurnGuard | Customer Churn Intelligence System</h1>
                <p class="hero-sub">
                    Executive-Grade Attrition Risk Forecasting, Real-Time Feature Engineering & Prescriptive Retention Playbooks
                </p>
            </div>
            <div style="margin-top: 10px;">
                <span style="background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.35); color: #c7d2fe; padding: 6px 14px; border-radius: 20px; font-size: 0.82rem; font-weight: 600;">
                    🚀 Model: {selected_model_name}
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab_prediction, tab_benchmark, tab_batch, tab_eda = st.tabs([
    "🔮 Live Churn Prediction & What-If Studio",
    "📊 Model Performance & Benchmarks",
    "📁 Batch Scoring (CSV)",
    "📑 Exploratory Insights (EDA)"
])

# =========================================================
# TAB 1: Live Churn Prediction & What-If Studio
# =========================================================
with tab_prediction:
    # -----------------------------------------------------
    # Top-Row KPI Highlights: The 3 Domain-Engineered Features
    # -----------------------------------------------------
    st.markdown("### 💡 Domain-Engineered Feature Signals (Top-Row KPIs)")
    st.caption("These 3 custom features are calculated dynamically from your inputs and passed directly into the preprocessor pipeline.")

    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

    spend_per_tenure_val = customer_df["Spend_per_Tenure"].iloc[0]
    support_intensity_val = customer_df["Support_Intensity"].iloc[0]
    high_delay_flag_val = int(customer_df["High_Delay_Flag"].iloc[0])

    with col_kpi1:
        # Value Velocity Card
        if spend_per_tenure_val > 40.0:
            kpi1_status = '<span style="color:#10b981; font-weight:700;">🟢 High Value Velocity</span>'
        elif spend_per_tenure_val > 20.0:
            kpi1_status = '<span style="color:#f59e0b; font-weight:700;">🟡 Moderate Commitment</span>'
        else:
            kpi1_status = '<span style="color:#ef4444; font-weight:700;">🔴 Low Capital Velocity</span>'

        st.markdown(
            f"""
            <div class="kpi-card" title="Formula: Total Spend / (Tenure + 1). Normalizes customer spend across account age.">
                <div>
                    <div class="kpi-title">Value Velocity</div>
                    <div class="kpi-value">${spend_per_tenure_val:.2f} <span style="font-size: 0.9rem; font-weight: 500; color: #94a3b8;">/ mo</span></div>
                    <div>{kpi1_status}</div>
                </div>
                <div>
                    <span class="kpi-formula-pill">Formula: Total Spend / (Tenure + 1)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_kpi2:
        # Friction Density Card
        if support_intensity_val > 0.35:
            kpi2_status = '<span style="color:#ef4444; font-weight:700;">🔴 Critical Support Friction</span>'
        elif support_intensity_val > 0.15:
            kpi2_status = '<span style="color:#f59e0b; font-weight:700;">🟡 Elevated Complaints</span>'
        else:
            kpi2_status = '<span style="color:#10b981; font-weight:700;">🟢 Smooth Experience</span>'

        st.markdown(
            f"""
            <div class="kpi-card" title="Formula: Support Calls / (Usage Frequency + 1). Renders complaints relative to engagement.">
                <div>
                    <div class="kpi-title">Friction Density</div>
                    <div class="kpi-value">{support_intensity_val:.2f} <span style="font-size: 0.9rem; font-weight: 500; color: #94a3b8;">calls/session</span></div>
                    <div>{kpi2_status}</div>
                </div>
                <div>
                    <span class="kpi-formula-pill">Formula: Support Calls / (Usage + 1)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_kpi3:
        # Billing Distress Risk Card
        if high_delay_flag_val == 1:
            kpi3_val_text = "RISK TRIGGERED"
            kpi3_color = "#ef4444"
            kpi3_status = '<span style="color:#ef4444; font-weight:700;">🔴 Overdue &gt; 15 Days</span>'
        else:
            kpi3_val_text = "NORMAL BILLING"
            kpi3_color = "#10b981"
            kpi3_status = '<span style="color:#10b981; font-weight:700;">🟢 On-Time / Minor (≤ 15 Days)</span>'

        st.markdown(
            f"""
            <div class="kpi-card" title="Formula: 1 if Payment Delay > 15 days else 0. Empirical threshold discovered in EDA.">
                <div>
                    <div class="kpi-title">Billing Distress Risk</div>
                    <div class="kpi-value" style="color: {kpi3_color}; font-size: 1.45rem; padding-top: 5px;">{kpi3_val_text}</div>
                    <div>{kpi3_status}</div>
                </div>
                <div>
                    <span class="kpi-formula-pill">Formula: 1 if Delay &gt; 15 else 0</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # -----------------------------------------------------
    # Middle Row: Visual Gauge & Executive Assessment
    # -----------------------------------------------------
    col_gauge, col_assessment = st.columns([1.15, 1.25])

    # Categorize Risk
    if churn_prob < 0.30:
        risk_level = "LOW RISK (SAFE)"
        badge_class = "status-badge-low"
        risk_icon = "🟢"
        headline = "Healthy Account: High Likelihood of Long-Term Retention"
        playbook_type = "safe"
    elif churn_prob < 0.70:
        risk_level = "MODERATE RISK"
        badge_class = "status-badge-med"
        risk_icon = "🟡"
        headline = "Vulnerable Account: Retention Countermeasures Advised"
        playbook_type = "warning"
    else:
        risk_level = "CRITICAL CHURN RISK"
        badge_class = "status-badge-high"
        risk_icon = "🔴"
        headline = "Severe Attrition Alert: Immediate Success Intervention Required"
        playbook_type = "danger"

    with col_gauge:
        st.markdown("#### 🧭 Visual Risk Gauge")
        gauge_fig = create_churn_gauge_chart(churn_prob)
        st.plotly_chart(gauge_fig, width="stretch", config={'displayModeBar': False})

    with col_assessment:
        st.markdown("#### 📋 Executive Risk Summary")
        st.markdown(f'<div class="{badge_class}">{risk_icon} {risk_level}</div>', unsafe_allow_html=True)
        st.markdown(f"### {headline}")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(
                label="Churn Propensity",
                value=f"{churn_prob * 100:.2f}%",
                delta=f"{'+' if churn_prob > 0.5 else '-'}{abs(churn_prob - 0.5)*100:.1f}% vs baseline",
                delta_color="inverse"
            )
        with col_m2:
            st.metric(
                label="Retention Likelihood",
                value=f"{retention_prob * 100:.2f}%",
                delta=f"{'+' if retention_prob > 0.5 else '-'}{abs(retention_prob - 0.5)*100:.1f}% vs baseline",
                delta_color="normal"
            )

        # Profile Snapshot Mini-Table
        summary_df = pd.DataFrame({
            "Core Parameter": ["Contract", "Subscription Tier", "Tenure", "Total Spend", "Support Calls", "Payment Delay"],
            "Configured Value": [
                f"{contract_length}", f"{subscription_type}", f"{tenure} months",
                f"${total_spend:,.2f}", f"{support_calls} calls logged", f"{payment_delay} days overdue"
            ]
        })
        st.dataframe(summary_df, hide_index=True, width="stretch")

    st.markdown("---")

    # -----------------------------------------------------
    # Prescriptive Action Playbook
    # -----------------------------------------------------
    st.markdown("### 🎯 Prescriptive Retention Playbook")
    st.markdown("Data-driven business interventions tailored specifically to this customer's friction profile:")

    if churn_prob >= 0.30:
        col_actions_a, col_actions_b = st.columns(2)

        with col_actions_a:
            if payment_delay > 15:
                st.markdown(
                    """
                    <div class="playbook-card playbook-card-danger">
                        <strong>💳 Delinquent Billing Intervention (Delay > 15 Days):</strong>
                        Customer has exceeded the critical payment delay threshold. Immediately dispatch an automated
                        grace-period extension, waive late penalty fees, and offer automated ACH monthly billing incentives.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            if contract_length == "Monthly":
                st.markdown(
                    """
                    <div class="playbook-card playbook-card-danger">
                        <strong>📄 Annual Contract Conversion Incentive:</strong>
                        Historical analysis reveals Monthly contract customers experience near-total attrition under friction.
                        Offer an Annual Contract upgrade incentive featuring a 20% loyalty discount to secure commitments.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col_actions_b:
            if support_calls >= 4:
                st.markdown(
                    """
                    <div class="playbook-card playbook-card-warning">
                        <strong>🎧 High Support Friction Alert (≥ 4 Calls):</strong>
                        Customer has escalated support issues multiple times. Assign a Dedicated Customer Success Manager
                        (CSM) to conduct a 24-hour white-glove review of outstanding support tickets.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            if last_interaction > 20:
                st.markdown(
                    """
                    <div class="playbook-card playbook-card-warning">
                        <strong>📢 Inactivity Re-Engagement:</strong>
                        Customer has not interacted with the platform in over 20 days. Initiate an automated re-onboarding
                        sequence highlighting newly shipped features and personalized platform tips.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        col_vip_1, col_vip_2 = st.columns(2)
        with col_vip_1:
            st.markdown(
                """
                <div class="playbook-card playbook-card-success">
                    <strong>🌟 VIP Loyalty Engagement:</strong>
                    Customer exhibits stellar retention stability. Enroll in exclusive Beta features, invite to customer
                    advisory councils, and offer referral bonus rewards to drive word-of-mouth acquisition.
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_vip_2:
            if subscription_type != "Premium":
                st.markdown(
                    """
                    <div class="playbook-card" style="border-left-color: #6366f1;">
                        <strong>🚀 High-Propensity Upsell Opportunity:</strong>
                        Zero detected friction indicates readiness for an upgrade to the <strong>Premium Tier</strong>.
                        Trigger targeted in-app messaging showcasing advanced analytics and unlimited seats.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # -----------------------------------------------------
    # Interactive Live 'What-If' Simulation Studio
    # -----------------------------------------------------
    st.markdown("---")
    st.markdown("### ⚡ Live 'What-If' Retention Simulation Studio")
    st.markdown(
        "Simulate corporate retention interventions in real time to observe how specific customer success "
        "actions directly reduce churn probability."
    )

    with st.expander("🛠️ Open Retention Levers & What-If Simulator", expanded=(churn_prob >= 0.30)):
        st.markdown(
            """
            <div class="whatif-box">
                <h4 style="margin-top:0; color: #818cf8;">Adjust Retention Levers for this Customer:</h4>
                <p style="font-size: 0.88rem; color: #cbd5e1;">
                    See how switching the contract length, resolving open support tickets, or extending billing relief alters the forecast:
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_w1, col_w2, col_w3 = st.columns(3)

        with col_w1:
            sim_contract = st.selectbox(
                "Simulated Contract:",
                ["Annual", "Quarterly", "Monthly"],
                index=0 if contract_length != "Annual" else 0,
                key="sim_contract",
                help="Test the effect of upgrading the customer to an Annual or Quarterly plan."
            )

        with col_w2:
            sim_calls = st.slider(
                "Simulated Support Calls (Post-Resolution):",
                min_value=0,
                max_value=int(max(support_calls, 1)),
                value=int(min(support_calls, 1)),
                key="sim_calls",
                help="Simulate the resolution of complaints through CSM support."
            )

        with col_w3:
            sim_delay = st.slider(
                "Simulated Payment Delay (Days Overdue):",
                min_value=0,
                max_value=int(max(payment_delay, 1)),
                value=int(min(payment_delay, 3)),
                key="sim_delay",
                help="Simulate restructuring late payments or forgiving overdue balances."
            )

        # Build simulated customer profile
        sim_raw = customer_raw.copy()
        sim_raw["Contract Length"] = sim_contract
        sim_raw["Support Calls"] = sim_calls
        sim_raw["Payment Delay"] = sim_delay

        sim_df = compute_engineered_features(sim_raw)
        _, sim_prob = predict_churn(sim_df, active_model)
        risk_reduction = (churn_prob - sim_prob) * 100

        st.markdown("#### 🔬 Simulation Results vs Baseline")
        col_res1, col_res2, col_res3 = st.columns(3)

        with col_res1:
            st.metric(
                label="Baseline Churn Propensity",
                value=f"{churn_prob * 100:.2f}%",
                help="Original predicted probability based on current profile."
            )

        with col_res2:
            st.metric(
                label="Simulated Churn Propensity",
                value=f"{sim_prob * 100:.2f}%",
                delta=f"-{risk_reduction:.2f}% Risk Reduction" if risk_reduction > 0 else f"+{abs(risk_reduction):.2f}%",
                delta_color="inverse"
            )

        with col_res3:
            if sim_prob < 0.30:
                sim_badge = '<span class="status-badge-low">🟢 NEW STATUS: SAFE (RETAINED)</span>'
            elif sim_prob < 0.70:
                sim_badge = '<span class="status-badge-med">🟡 NEW STATUS: MODERATE RISK</span>'
            else:
                sim_badge = '<span class="status-badge-high">🔴 NEW STATUS: STILL CRITICAL</span>'
            st.markdown("<div style='padding-top: 10px;'>Simulated Customer Status:</div>", unsafe_allow_html=True)
            st.markdown(sim_badge, unsafe_allow_html=True)

        if risk_reduction > 10.0:
            st.success(
                f"🎉 **High-Impact Intervention!** Applying these retention levers reduces customer churn probability by "
                f"**{risk_reduction:.1f}%**, de-risking the account and preserving lifetime recurring revenue."
            )
        elif risk_reduction > 0:
            st.info(f"Intervention produces a moderate risk decrease of **{risk_reduction:.1f}%**.")
        else:
            st.caption("No risk reduction observed under the selected parameters.")

# =========================================================
# TAB 2: Model Performance & Benchmarks
# =========================================================
with tab_benchmark:
    st.markdown("## 📊 Machine Learning Model Benchmarking & Diagnostics")
    st.markdown(
        "Four candidate algorithms were trained on **352,665** customer records and evaluated on **88,167** stratified "
        "holdout samples to establish rigorous comparative metrics."
    )

    if benchmark_data and "models" in benchmark_data:
        benchmark_rows = []
        for model_name, info in benchmark_data["models"].items():
            val = info["validation_metrics"]
            is_champion = "🏆 Champion" if model_name == benchmark_data.get("champion_model") else "Candidate"
            benchmark_rows.append({
                "Algorithm": model_name,
                "Status": is_champion,
                "Train Time (s)": f"{info['train_time_seconds']:.2f}s",
                "Val Accuracy": f"{val['accuracy']*100:.2f}%",
                "Val Precision": f"{val['precision']*100:.2f}%",
                "Val Recall": f"{val['recall']*100:.2f}%",
                "Val F1-Score": f"{val['f1']:.4f}",
                "Val ROC-AUC": f"{val['roc_auc']:.4f}"
            })
        df_bench = pd.DataFrame(benchmark_rows)
        st.dataframe(df_bench, hide_index=True, width="stretch")
    else:
        st.info("Execute `python scripts/04_train_models.py` to regenerate the dynamic benchmark matrix.")

    st.markdown("---")
    st.markdown("### 📈 Visual Benchmark & Diagnostic Artifacts")

    col_chart_1, col_chart_2 = st.columns(2)

    chart_metrics = os.path.join(CHARTS_DIR, "03_model_metrics_comparison.png")
    chart_cm = os.path.join(CHARTS_DIR, "04_confusion_matrices.png")
    chart_fi = os.path.join(CHARTS_DIR, "05_feature_importance.png")

    with col_chart_1:
        if os.path.exists(chart_metrics):
            st.image(chart_metrics, caption="Figure 3: Multi-Metric Benchmark Comparison across Algorithmic Families")
        else:
            st.warning("Chart 03_model_metrics_comparison.png not found.")

    with col_chart_2:
        if os.path.exists(chart_cm):
            st.image(chart_cm, caption="Figure 4: Confusion Matrices for All 4 Candidate Classifiers")
        else:
            st.warning("Chart 04_confusion_matrices.png not found.")

    st.markdown("---")
    st.markdown("### 🔍 Model Explainability & Key Churn Drivers")

    col_fi_img, col_fi_desc = st.columns([1.1, 1])

    with col_fi_img:
        if os.path.exists(chart_fi):
            st.image(chart_fi, caption="Figure 5: Feature Importances (Gini Impurity from Random Forest)")
        else:
            st.warning("Chart 05_feature_importance.png not found.")

    with col_fi_desc:
        st.markdown(
            """
            #### Key Data Science Takeaways:
            1. **Contract Structure Dominance**:
               Monthly contract agreements represent the strongest single predictor of churn. Without contractual locking, even minor operational hurdles trigger immediate cancellation.
            2. **Support Escalation Threshold**:
               Customer service calls exhibit a non-linear effect. Accounts logging 4 or more calls cross an inflection point where dissatisfaction overrides general platform utility.
            3. **Payment Delinquency Risk**:
               Delays greater than 15 days serve as a reliable early indicator of subscription abandonment or payment failure.
            4. **Engineered Feature Impact**:
               The engineered features (`Spend_per_Tenure` and `Support_Intensity`) significantly sharpen decision boundaries across both linear and tree-based estimators.
            """
        )

# =========================================================
# TAB 3: Batch Customer Scoring (CSV)
# =========================================================
with tab_batch:
    st.markdown("## 📁 Enterprise Batch Customer Scoring")
    st.markdown(
        "Upload a CSV file containing multiple customer accounts to execute automated bulk feature engineering "
        "and generate predictive churn risk classifications at scale."
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

    uploaded_file = st.file_uploader("Upload customer CSV file to score:", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded **{len(batch_df):,}** customer records successfully.")

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
                    bins=[-0.01, 0.30, 0.70, 1.0],
                    labels=["Low Risk (Safe)", "Moderate Risk", "Critical Risk"]
                )

                st.markdown("### 📊 Batch Scoring Results Summary")
                col_b1, col_b2, col_b3 = st.columns(3)
                high_risk_count = int((scored_df["Risk_Category"] == "Critical Risk").sum())
                med_risk_count = int((scored_df["Risk_Category"] == "Moderate Risk").sum())
                low_risk_count = int((scored_df["Risk_Category"] == "Low Risk (Safe)").sum())

                with col_b1:
                    st.metric("Critical Risk Accounts", f"{high_risk_count:,}", f"{high_risk_count/len(scored_df)*100:.1f}%")
                with col_b2:
                    st.metric("Moderate Risk Accounts", f"{med_risk_count:,}", f"{med_risk_count/len(scored_df)*100:.1f}%")
                with col_b3:
                    st.metric("Safe Retained Accounts", f"{low_risk_count:,}", f"{low_risk_count/len(scored_df)*100:.1f}%")

                display_cols = [c for c in ["CustomerID", "Age", "Gender", "Contract Length",
                                           "Total Spend", "Churn_Probability", "Risk_Category"] if c in scored_df.columns]
                st.dataframe(scored_df[display_cols], width="stretch")

                # Export scored CSV
                out_buffer = io.StringIO()
                scored_df.to_csv(out_buffer, index=False)
                st.download_button(
                    label="💾 Export Enriched Scored CSV",
                    data=out_buffer.getvalue(),
                    file_name="scored_customer_churn_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")

# =========================================================
# TAB 4: Exploratory Data Analysis (EDA) Insights
# =========================================================
with tab_eda:
    st.markdown("## 📑 Exploratory Data Analysis (EDA) Findings")
    st.markdown(
        "Key empirical discoveries and behavioral distributions uncovered during dataset exploration (`scripts/02_data_analysis.py`)."
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
        ### 💡 Strategic Commercial Takeaways:
        - **Monthly Contracts vs Annual Commitments**: Customers on monthly billing cancel at a ~100% rate upon facing product friction, whereas annual subscriptions build a strong barrier against churn.
        - **Support Friction Points**: Churned customers average **4.92 support calls** compared to **1.81 calls** for retained customers (+172% increase). Interventions must trigger upon the 3rd logged call.
        - **Payment Delinquency**: Delays beyond 15 days exhibit over 85% conversion to cancellation, making overdue billing the single most urgent churn trigger.
        """
    )

# ---------------------------------------------------------
# Executive Footer
# ---------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        <strong>ChurnGuard Intelligence System</strong> • Developed by 
        <a href="https://linkedin.com/in/kalindusilva" target="_blank">Kalindu Mihiranga Silva</a><br>
        2nd-Year BSc (Hons) Data Science & Business Analytics • General Sir John Kotelawala Defence University (KDU), Sri Lanka<br>
        Repository: <a href="https://github.com/kalinduusilva/customer-churn-prediction" target="_blank">github.com/kalinduusilva/customer-churn-prediction</a>
    </div>
    """,
    unsafe_allow_html=True
)
