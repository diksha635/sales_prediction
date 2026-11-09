"""
Sales Prediction Application
============================
A modern, interactive Streamlit application for predicting sales revenue based on
multi-channel advertising expenditures (TV, Radio, Newspaper) using Scikit-Learn's
Multiple Linear Regression.

Project Structure:
- Framework: Streamlit
- ML Library: scikit-learn (LinearRegression, train_test_split, evaluation metrics)
- Visualization: matplotlib, seaborn
- Data Processing: pandas, numpy

Author: Advanced Agentic Engineer
"""

import os
import tempfile
from pathlib import Path

# Use temporary directory for matplotlib cache on Windows to avoid permission issues
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION & MODERN STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sales Prediction & Ad Budget Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern UI cards, badges, and typography
st.markdown(
    """
    <style>
    /* Global Container Spacing */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 14px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.95rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.15;
    }
    .metric-sub {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 6px;
    }
    .highlight-card {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 1px solid #bfdbfe;
        border-left: 6px solid #2563eb;
    }
    .highlight-card .metric-value {
        color: #1d4ed8;
    }
    
    /* Channel input box card */
    .channel-box {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 12px;
        border-left: 4px solid #3b82f6;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-green { background-color: #dcfce7; color: #15803d; }
    .badge-blue { background-color: #dbeafe; color: #1d4ed8; }
    .badge-amber { background-color: #fef3c7; color: #b45309; }

    /* Section header styling */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 14px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. CONSTANTS
# -----------------------------------------------------------------------------
FEATURES = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"
DATA_FILE_DEFAULT = "advertising.csv"


# -----------------------------------------------------------------------------
# 3. DATA LOADING & SYNTHETIC DATA INITIALIZATION
# -----------------------------------------------------------------------------
def generate_synthetic_data(n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate realistic dummy training data matching standard advertising benchmarks.

    Ensures the model fits correctly on startup if no external dataset file is present.
    """
    rng = np.random.default_rng(seed)
    tv = rng.uniform(5.0, 300.0, n_samples)
    radio = rng.uniform(0.0, 50.0, n_samples)
    newspaper = rng.uniform(0.5, 115.0, n_samples)

    # Underlying linear relationship:
    # Sales ≈ 4.625 + 0.0544 * TV + 0.1070 * Radio + 0.0003 * Newspaper + Normal Noise
    noise = rng.normal(0.0, 1.55, n_samples)
    sales = 4.625 + (0.0544 * tv) + (0.1070 * radio) + (0.0003 * newspaper) + noise
    sales = np.clip(sales, a_min=1.0, a_max=None)

    df = pd.DataFrame(
        {
            "TV": np.round(tv, 1),
            "Radio": np.round(radio, 1),
            "Newspaper": np.round(newspaper, 1),
            "Sales": np.round(sales, 1),
        }
    )
    return df


def load_dataset(source_type: str, uploaded_file=None) -> tuple[pd.DataFrame, str]:
    """Load data from local file, uploaded file, or synthetic generator."""
    # 1. Custom Upload
    if source_type == "Upload Custom CSV" and uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            lower_map = {col.lower(): col for col in df.columns}
            needed = ["tv", "radio", "newspaper", "sales"]
            missing = [n for n in needed if n not in lower_map]
            if missing:
                return (
                    generate_synthetic_data(),
                    f"Uploaded CSV missing required columns: {missing}. Using fallback sample data.",
                )
            rename_dict = {
                lower_map["tv"]: "TV",
                lower_map["radio"]: "Radio",
                lower_map["newspaper"]: "Newspaper",
                lower_map["sales"]: "Sales",
            }
            df = df.rename(columns=rename_dict)
            df = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
            return df, f"Successfully loaded {len(df)} rows from uploaded file '{uploaded_file.name}'."
        except Exception as err:
            return (
                generate_synthetic_data(),
                f"Error parsing uploaded file ({err}). Using fallback synthetic data.",
            )

    # 2. Synthetic Dummy Data Option
    if source_type == "Synthetic Sample Data":
        return generate_synthetic_data(), "Initialized with synthetic sample dataset (200 records)."

    # 3. Default: Search local filesystem
    search_paths = [
        Path(__file__).resolve().parent / DATA_FILE_DEFAULT,
        Path(__file__).resolve().parent / "Advertising.csv",
        Path(DATA_FILE_DEFAULT),
    ]

    for p in search_paths:
        if p.exists():
            try:
                df = pd.read_csv(p)
                df.columns = df.columns.str.strip()
                valid_cols = [c for c in df.columns if c in FEATURES + [TARGET]]
                df = df[valid_cols].apply(pd.to_numeric, errors="coerce").dropna()
                if not df.empty and set(FEATURES + [TARGET]).issubset(df.columns):
                    return df, f"Loaded {len(df)} records from local dataset '{p.name}'."
            except Exception:
                pass

    # Automatic fallback if file is not found
    synthetic_df = generate_synthetic_data()
    return (
        synthetic_df,
        "No local advertising.csv found. Automatically initialized realistic sample dataset.",
    )


# -----------------------------------------------------------------------------
# 4. MODEL TRAINING & PREDICTION PIPELINE
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def train_linear_regression(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """Train Multiple Linear Regression model with scikit-learn and compute metrics."""
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    r2_test = r2_score(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)

    # Adjusted R²
    n = len(y_test)
    p = X.shape[1]
    adj_r2_test = 1.0 - (1.0 - r2_test) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else r2_test

    coef_dict = dict(zip(FEATURES, model.coef_))
    intercept = float(model.intercept_)

    return {
        "model": model,
        "intercept": intercept,
        "coef": coef_dict,
        "metrics": {
            "r2_test": r2_test,
            "r2_train": r2_train,
            "adj_r2_test": adj_r2_test,
            "mae_test": mae_test,
            "mse_test": mse_test,
            "rmse_test": rmse_test,
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
    }


def predict_revenue(model: LinearRegression, tv: float, radio: float, newspaper: float) -> float:
    """Predict sales volume from channel advertising spend."""
    sample_df = pd.DataFrame([[tv, radio, newspaper]], columns=FEATURES)
    return float(model.predict(sample_df)[0])


# -----------------------------------------------------------------------------
# 5. SYNCHRONIZATION CALLBACKS & SESSION STATE
# -----------------------------------------------------------------------------
# Default spend values
if "tv_slider" not in st.session_state:
    st.session_state.tv_slider = 150.0
if "tv_num" not in st.session_state:
    st.session_state.tv_num = 150.0

if "radio_slider" not in st.session_state:
    st.session_state.radio_slider = 25.0
if "radio_num" not in st.session_state:
    st.session_state.radio_num = 25.0

if "news_slider" not in st.session_state:
    st.session_state.news_slider = 30.0
if "news_num" not in st.session_state:
    st.session_state.news_num = 30.0

if "scenarios" not in st.session_state:
    st.session_state.scenarios = []

if "predict_triggered" not in st.session_state:
    st.session_state.predict_triggered = False


# Two-way sync callbacks between sliders and number inputs
def sync_tv_slider():
    st.session_state.tv_num = float(st.session_state.tv_slider)


def sync_tv_num():
    st.session_state.tv_slider = float(min(st.session_state.tv_num, 350.0))


def sync_radio_slider():
    st.session_state.radio_num = float(st.session_state.radio_slider)


def sync_radio_num():
    st.session_state.radio_slider = float(min(st.session_state.radio_num, 60.0))


def sync_news_slider():
    st.session_state.news_num = float(st.session_state.news_slider)


def sync_news_num():
    st.session_state.news_slider = float(min(st.session_state.news_num, 120.0))


def apply_campaign_preset(tv_val, radio_val, news_val):
    """Update all inputs and trigger prediction immediately."""
    st.session_state.tv_slider = float(tv_val)
    st.session_state.tv_num = float(tv_val)
    st.session_state.radio_slider = float(radio_val)
    st.session_state.radio_num = float(radio_val)
    st.session_state.news_slider = float(news_val)
    st.session_state.news_num = float(news_val)
    st.session_state.predict_triggered = True


# -----------------------------------------------------------------------------
# 6. SIDEBAR CONTROLS & NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 SalesPredict AI")
    st.caption("Marketing Mix Modeling with Linear Regression")
    st.markdown("---")

    nav_choice = st.radio(
        "Navigation",
        [
            "🔮 Sales Predictor",
            "📈 Model Diagnostics",
            "📊 Data Explorer & EDA",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📁 Data Source")
    data_source_mode = st.selectbox(
        "Dataset Source",
        ["Default Dataset (Local)", "Synthetic Sample Data", "Upload Custom CSV"],
        index=0,
    )

    custom_file = None
    if data_source_mode == "Upload Custom CSV":
        custom_file = st.file_uploader("Upload CSV (TV, Radio, Newspaper, Sales)", type=["csv"])

    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    test_ratio = st.slider(
        "Test Split Size",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.05,
        help="Proportion of dataset reserved for testing.",
    )
    seed_val = st.number_input(
        "Random Seed",
        min_value=1,
        max_value=9999,
        value=42,
        step=1,
    )

    st.markdown("---")
    st.markdown("### ⚡ Campaign Presets")
    col_pre1, col_pre2 = st.columns(2)
    with col_pre1:
        if st.button("⚖️ Balanced", use_container_width=True):
            apply_campaign_preset(150.0, 25.0, 25.0)
            st.rerun()
        if st.button("📻 Radio Boost", use_container_width=True):
            apply_campaign_preset(60.0, 48.0, 15.0)
            st.rerun()
    with col_pre2:
        if st.button("📺 TV Blitz", use_container_width=True):
            apply_campaign_preset(280.0, 15.0, 10.0)
            st.rerun()
        if st.button("🌱 Low Budget", use_container_width=True):
            apply_campaign_preset(30.0, 12.0, 8.0)
            st.rerun()

# Train Model on startup / selection
dataset, status_message = load_dataset(data_source_mode, custom_file)
results = train_linear_regression(dataset, test_size=test_ratio, random_state=seed_val)
model = results["model"]
metrics = results["metrics"]
coefs = results["coef"]
intercept = results["intercept"]

# Current active spend values
tv_val = float(st.session_state.tv_num)
radio_val = float(st.session_state.radio_num)
newspaper_val = float(st.session_state.news_num)
total_spend = tv_val + radio_val + newspaper_val

# -----------------------------------------------------------------------------
# 7. PAGE 1: SALES PREDICTOR & SCENARIO ANALYSIS
# -----------------------------------------------------------------------------
if nav_choice == "🔮 Sales Predictor":
    st.title("🎯 Advertising Spend & Sales Revenue Prediction")
    st.write(
        "Adjust advertising budgets across **TV**, **Radio**, and **Newspaper** "
        "to estimate expected sales volume and evaluate marketing ROI."
    )

    st.info(f"📌 **Status:** {status_message}")

    # Section 1: Spend Inputs
    st.markdown('<div class="section-header">💼 1. Configure Advertising Budgets (in thousands of dollars)</div>', unsafe_allow_html=True)

    c_tv, c_radio, c_news = st.columns(3)

    # TV Controls
    with c_tv:
        st.markdown(
            """
            <div class="channel-box" style="border-left-color: #3b82f6;">
                <strong>📺 TV Advertising</strong><br>
                <span style="font-size:0.8rem; color:#64748b;">Broad visual reach & mass brand awareness</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.slider(
            "TV Budget Slider ($k)",
            min_value=0.0,
            max_value=350.0,
            step=1.0,
            key="tv_slider",
            on_change=sync_tv_slider,
        )
        st.number_input(
            "TV Exact Budget ($k)",
            min_value=0.0,
            max_value=1000.0,
            step=1.0,
            key="tv_num",
            on_change=sync_tv_num,
        )

    # Radio Controls
    with c_radio:
        st.markdown(
            """
            <div class="channel-box" style="border-left-color: #10b981;">
                <strong>📻 Radio Advertising</strong><br>
                <span style="font-size:0.8rem; color:#64748b;">High engagement & localized consumer action</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.slider(
            "Radio Budget Slider ($k)",
            min_value=0.0,
            max_value=60.0,
            step=0.5,
            key="radio_slider",
            on_change=sync_radio_slider,
        )
        st.number_input(
            "Radio Exact Budget ($k)",
            min_value=0.0,
            max_value=500.0,
            step=0.5,
            key="radio_num",
            on_change=sync_radio_num,
        )

    # Newspaper Controls
    with c_news:
        st.markdown(
            """
            <div class="channel-box" style="border-left-color: #f59e0b;">
                <strong>📰 Newspaper Advertising</strong><br>
                <span style="font-size:0.8rem; color:#64748b;">Traditional print targeting regional demographics</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.slider(
            "Newspaper Budget Slider ($k)",
            min_value=0.0,
            max_value=120.0,
            step=0.5,
            key="news_slider",
            on_change=sync_news_slider,
        )
        st.number_input(
            "Newspaper Exact Budget ($k)",
            min_value=0.0,
            max_value=500.0,
            step=0.5,
            key="news_num",
            on_change=sync_news_num,
        )

    # Budget Distribution Indicator
    if total_spend > 0:
        pct_tv = (tv_val / total_spend) * 100
        pct_radio = (radio_val / total_spend) * 100
        pct_news = (newspaper_val / total_spend) * 100
    else:
        pct_tv, pct_radio, pct_news = 0, 0, 0

    st.markdown(
        f"""
        <div style="background:#f1f5f9; padding:10px 14px; border-radius:8px; font-size:0.88rem; color:#334155; margin-top:8px;">
            <strong>Budget Allocation:</strong> 
            TV: <strong>{pct_tv:.1f}%</strong> (${tv_val:,.1f}k) &nbsp;|&nbsp; 
            Radio: <strong>{pct_radio:.1f}%</strong> (${radio_val:,.1f}k) &nbsp;|&nbsp; 
            Newspaper: <strong>{pct_news:.1f}%</strong> (${newspaper_val:,.1f}k) &nbsp;|&nbsp; 
            Total: <strong>${total_spend:,.1f}k</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Section: Prominent Action Button
    col_btn_l, col_btn_c, col_btn_r = st.columns([1, 2, 1])
    with col_btn_c:
        clicked_predict = st.button(
            "🚀 Predict Sales Revenue",
            type="primary",
            use_container_width=True,
        )
        if clicked_predict:
            st.session_state.predict_triggered = True

    # -------------------------------------------------------------------------
    # Predictions & Prominent Metric Cards
    # -------------------------------------------------------------------------
    predicted_val = predict_revenue(model, tv_val, radio_val, newspaper_val)
    mean_sales = dataset[TARGET].mean()
    delta_sales = predicted_val - mean_sales
    delta_pct = (delta_sales / mean_sales) * 100 if mean_sales > 0 else 0.0

    # 95% Confidence Interval using Test RMSE
    rmse = metrics["rmse_test"]
    ci_lower = max(0.0, predicted_val - 1.96 * rmse)
    ci_upper = predicted_val + 1.96 * rmse
    sales_to_spend = (predicted_val / total_spend) if total_spend > 0 else 0.0

    st.markdown("---")
    st.markdown('<div class="section-header">📊 2. Predicted Sales & Performance Metrics</div>', unsafe_allow_html=True)

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.markdown(
            f"""
            <div class="metric-card highlight-card">
                <div class="metric-title">🎯 Predicted Sales Revenue</div>
                <div class="metric-value">{predicted_val:.2f} <span style="font-size:1rem; font-weight:normal;">k units</span></div>
                <div class="metric-sub">
                    <span class="badge {'badge-green' if delta_sales >= 0 else 'badge-amber'}">
                        {'+' if delta_sales >= 0 else ''}{delta_sales:.2f} ({delta_pct:+.1f}%) vs historical avg
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">💰 Total Advertising Budget</div>
                <div class="metric-value">${total_spend:.1f} <span style="font-size:1rem; font-weight:normal; color:#64748b;">k</span></div>
                <div class="metric-sub">
                    Spread across 3 marketing channels
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">📈 95% Prediction Interval</div>
                <div class="metric-value" style="font-size:1.45rem;">[{ci_lower:.2f} - {ci_upper:.2f}]</div>
                <div class="metric-sub">± {1.96 * rmse:.2f} k units (based on RMSE)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">⚡ Sales-to-Spend Multiplier</div>
                <div class="metric-value">{sales_to_spend:.2f}x</div>
                <div class="metric-sub">Baseline Organic Sales: {intercept:.2f} k</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Visualizations: Feature Contributions & Regression Lines
    # -------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📉 3. Interactive Visualizations & Attribution</div>', unsafe_allow_html=True)

    tab_contrib, tab_regression = st.tabs(
        [
            "🧩 Feature Contribution Breakdown",
            "📈 Regression Lines & Current Scenario",
        ]
    )

    with tab_contrib:
        # Calculate feature contributions
        contrib_tv = coefs["TV"] * tv_val
        contrib_radio = coefs["Radio"] * radio_val
        contrib_news = coefs["Newspaper"] * newspaper_val

        breakdown_df = pd.DataFrame(
            {
                "Component": [
                    "Baseline Sales (Zero Ad Spend)",
                    "TV Spend Contribution",
                    "Radio Spend Contribution",
                    "Newspaper Spend Contribution",
                    "Total Predicted Sales",
                ],
                "Sales Units (k)": [
                    intercept,
                    contrib_tv,
                    contrib_radio,
                    contrib_news,
                    predicted_val,
                ],
            }
        )

        fig_bar, ax_bar = plt.subplots(figsize=(9.5, 4.4))
        bar_colors = ["#94a3b8", "#3b82f6", "#10b981", "#f59e0b", "#4f46e5"]
        bars = ax_bar.barh(
            breakdown_df["Component"],
            breakdown_df["Sales Units (k)"],
            color=bar_colors,
            edgecolor="#0f172a",
            linewidth=0.7,
            height=0.55,
        )

        # Annotate exact numbers on each bar
        for bar, val in zip(bars, breakdown_df["Sales Units (k)"]):
            x_pos = val + 0.35 if val >= 0 else 0.35
            ax_bar.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f} k",
                va="center",
                ha="left",
                fontsize=9.5,
                fontweight="bold",
                color="#0f172a",
            )

        ax_bar.set_xlabel("Sales Units Contribution (in thousands)", fontsize=10, fontweight="bold")
        ax_bar.set_title(
            "Contribution Breakdown: How Marketing Spend Builds Total Sales",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax_bar.grid(axis="x", linestyle="--", alpha=0.5)
        ax_bar.set_xlim(0, max(predicted_val * 1.25, 20))
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

        st.caption(
            f"**Linear Model Formula:** Sales = {intercept:.4f} + "
            f"({coefs['TV']:.4f} × TV) + ({coefs['Radio']:.4f} × Radio) + ({coefs['Newspaper']:.4f} × Newspaper)"
        )

    with tab_regression:
        # Plot 3 side-by-side marginal regression plots with current point
        fig_regr, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        channel_meta = [
            ("TV", tv_val, "#3b82f6", 0),
            ("Radio", radio_val, "#10b981", 1),
            ("Newspaper", newspaper_val, "#f59e0b", 2),
        ]

        for channel, u_val, col, idx in channel_meta:
            ax = axes[idx]
            # Historical dataset points
            ax.scatter(
                dataset[channel],
                dataset[TARGET],
                color="#94a3b8",
                alpha=0.5,
                edgecolors="none",
                s=32,
                label="Historical Data",
            )

            # Fit marginal trend line
            x_fit = np.linspace(0, dataset[channel].max() * 1.15, 100)
            m_single = LinearRegression().fit(dataset[[channel]], dataset[TARGET])
            y_fit = m_single.predict(pd.DataFrame({channel: x_fit}))
            ax.plot(
                x_fit,
                y_fit,
                color=col,
                linewidth=2.2,
                label=f"Trend (slope={m_single.coef_[0]:.3f})",
            )

            # Highlight current user spend marker
            ax.scatter(
                [u_val],
                [predicted_val],
                color="#ef4444",
                s=140,
                marker="*",
                edgecolor="black",
                linewidth=1,
                zorder=6,
                label=f"Current: ${u_val:.0f}k",
            )

            ax.set_title(f"{channel} Spend vs. Sales", fontsize=11, fontweight="bold")
            ax.set_xlabel(f"{channel} Budget ($k)", fontsize=9.5)
            ax.set_ylabel("Sales Units (k)" if idx == 0 else "", fontsize=9.5)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper left", fontsize=8)

        plt.suptitle(
            "Marginal Regression Lines with Current Scenario Marker (★)",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        st.pyplot(fig_regr)
        plt.close(fig_regr)

    # -------------------------------------------------------------------------
    # Scenario Comparison Section
    # -------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">💾 4. Scenario Comparison & Bookmarking</div>', unsafe_allow_html=True)

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        sc_name = st.text_input(
            "Scenario Name",
            value=f"Scenario #{len(st.session_state.scenarios) + 1}",
        )
    with col_s2:
        st.write("")
        st.write("")
        if st.button("➕ Save Current Scenario", use_container_width=True):
            st.session_state.scenarios.append(
                {
                    "Scenario": sc_name,
                    "TV ($k)": tv_val,
                    "Radio ($k)": radio_val,
                    "Newspaper ($k)": newspaper_val,
                    "Total Budget ($k)": total_spend,
                    "Predicted Sales (k)": round(predicted_val, 2),
                    "Sales/Spend Multiplier": round(sales_to_spend, 2),
                }
            )
            st.success(f"Saved '{sc_name}'!")

    if st.session_state.scenarios:
        sc_table = pd.DataFrame(st.session_state.scenarios)
        st.dataframe(sc_table, use_container_width=True)

        sc_dl, sc_clr = st.columns([2, 1])
        with sc_dl:
            csv_sc = sc_table.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Scenarios as CSV",
                data=csv_sc,
                file_name="budget_scenarios.csv",
                mime="text/csv",
            )
        with sc_clr:
            if st.button("🗑️ Clear All Scenarios"):
                st.session_state.scenarios = []
                st.rerun()

# -----------------------------------------------------------------------------
# 8. PAGE 2: MODEL DIAGNOSTICS & PERFORMANCE
# -----------------------------------------------------------------------------
elif nav_choice == "📈 Model Diagnostics":
    st.title("📈 Model Evaluation & Diagnostic Insights")
    st.write(
        f"Detailed performance breakdown of the **Multiple Linear Regression** model trained on "
        f"**{metrics['n_train']}** samples and evaluated on **{metrics['n_test']}** hold-out test samples."
    )

    # Diagnostic Metric Cards
    diag1, diag2, diag3, diag4 = st.columns(4)
    with diag1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Test R² Score</div>
                <div class="metric-value">{metrics['r2_test']:.4f}</div>
                <div class="metric-sub">Train R²: {metrics['r2_train']:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with diag2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">RMSE (Root Mean Sq Error)</div>
                <div class="metric-value">{metrics['rmse_test']:.3f}</div>
                <div class="metric-sub">Average prediction error</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with diag3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">MAE (Mean Absolute Error)</div>
                <div class="metric-value">{metrics['mae_test']:.3f}</div>
                <div class="metric-sub">Mean absolute deviation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with diag4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Adjusted R² Score</div>
                <div class="metric-value">{metrics['adj_r2_test']:.4f}</div>
                <div class="metric-sub">Penalizes redundant features</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("📐 Fitted Multiple Linear Regression Equation")

    st.latex(
        r"\text{Sales} = \beta_0 + \beta_{\text{TV}} \cdot \text{TV} + \beta_{\text{Radio}} \cdot \text{Radio} + \beta_{\text{Newspaper}} \cdot \text{Newspaper}"
    )
    st.latex(
        rf"\widehat{{\text{{Sales}}}} = {intercept:.4f} + ({coefs['TV']:.4f} \cdot \text{{TV}}) + ({coefs['Radio']:.4f} \cdot \text{{Radio}}) + ({coefs['Newspaper']:.4f} \cdot \text{{Newspaper}})"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Coefficient Analysis & Business Interpretation")

    coef_df = pd.DataFrame(
        {
            "Predictor": ["TV Advertising", "Radio Advertising", "Newspaper Advertising", "Intercept (Baseline)"],
            "Coefficient (Weight)": [coefs["TV"], coefs["Radio"], coefs["Newspaper"], intercept],
            "Business Interpretation": [
                f"Every $1,000 spent on TV increases sales by approx. {coefs['TV']*1000:.1f} units.",
                f"Every $1,000 spent on Radio increases sales by approx. {coefs['Radio']*1000:.1f} units.",
                f"Every $1,000 spent on Newspaper increases sales by approx. {coefs['Newspaper']*1000:.1f} units.",
                "Estimated baseline sales without any advertising expenditure.",
            ],
        }
    )
    st.dataframe(coef_df, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Residual & Regression Diagnostic Plots")

    col_diag_l, col_diag_r = st.columns(2)

    with col_diag_l:
        # Actual vs Predicted Scatter
        y_test = results["y_test"]
        y_pred = results["y_test_pred"]

        fig_act, ax_act = plt.subplots(figsize=(6, 4.8))
        ax_act.scatter(y_test, y_pred, color="#4f46e5", alpha=0.75, edgecolors="white", s=65)

        # Ideal 45-degree line
        min_lim = min(y_test.min(), y_pred.min()) - 1
        max_lim = max(y_test.max(), y_pred.max()) + 1
        ax_act.plot([min_lim, max_lim], [min_lim, max_lim], "r--", linewidth=1.8, label="Ideal Fit (y = ŷ)")

        ax_act.set_xlabel("Actual Sales (k units)", fontsize=10, fontweight="bold")
        ax_act.set_ylabel("Predicted Sales (k units)", fontsize=10, fontweight="bold")
        ax_act.set_title(f"Actual vs. Predicted Sales (R² = {metrics['r2_test']:.3f})", fontsize=11, fontweight="bold")
        ax_act.grid(True, linestyle=":", alpha=0.6)
        ax_act.legend()
        plt.tight_layout()
        st.pyplot(fig_act)
        plt.close(fig_act)

    with col_diag_r:
        # Residuals Histogram & KDE
        residuals = y_test - y_pred

        fig_res, ax_res = plt.subplots(figsize=(6, 4.8))
        sns.histplot(residuals, kde=True, color="#0ea5e9", ax=ax_res, bins=12)
        ax_res.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error Line")
        ax_res.set_xlabel("Residual (Actual - Predicted)", fontsize=10, fontweight="bold")
        ax_res.set_ylabel("Frequency", fontsize=10, fontweight="bold")
        ax_res.set_title("Residual Error Distribution (Normality Check)", fontsize=11, fontweight="bold")
        ax_res.grid(True, linestyle=":", alpha=0.6)
        ax_res.legend()
        plt.tight_layout()
        st.pyplot(fig_res)
        plt.close(fig_res)

# -----------------------------------------------------------------------------
# 9. PAGE 3: DATA EXPLORER & EDA
# -----------------------------------------------------------------------------
elif nav_choice == "📊 Data Explorer & EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.write("Inspect dataset distributions, summary metrics, and feature correlations.")

    top_l, top_r = st.columns([3, 2])

    with top_l:
        st.subheader("📋 Dataset Inspection")
        st.dataframe(dataset, use_container_width=True, height=270)
        st.caption(f"Shape: {dataset.shape[0]} rows × {dataset.shape[1]} columns")

        dl_data = dataset.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Current Dataset as CSV",
            data=dl_data,
            file_name="advertising_dataset.csv",
            mime="text/csv",
        )

    with top_r:
        st.subheader("📈 Statistical Overview")
        st.dataframe(
            dataset.describe().T[["mean", "std", "min", "50%", "max"]],
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("🔥 Correlation Matrix & Spend Distributions")

    eda_l, eda_r = st.columns(2)

    with eda_l:
        fig_heat, ax_heat = plt.subplots(figsize=(5.5, 4.5))
        corr_matrix = dataset.corr()
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="Blues",
            fmt=".3f",
            linewidths=0.5,
            square=True,
            ax=ax_heat,
        )
        ax_heat.set_title("Pearson Correlation Matrix", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_heat)
        plt.close(fig_heat)

    with eda_r:
        fig_box, ax_box = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(
            data=dataset[FEATURES],
            palette=["#3b82f6", "#10b981", "#f59e0b"],
            ax=ax_box,
        )
        ax_box.set_ylabel("Expenditure ($k)", fontsize=10, fontweight="bold")
        ax_box.set_title("Advertising Spend Distributions", fontsize=11, fontweight="bold")
        ax_box.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig_box)
        plt.close(fig_box)

    st.info(
        "💡 **Key Observation:** TV demonstrates the highest correlation with Sales (r ≈ 0.90), "
        "followed by Radio (r ≈ 0.35). Newspaper shows very low direct correlation, confirming that "
        "reallocating budget towards TV and Radio typically maximizes revenue."
    )
