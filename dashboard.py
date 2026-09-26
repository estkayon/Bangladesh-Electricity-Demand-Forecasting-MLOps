from textwrap import dedent

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Bangladesh Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Helper
# --------------------------------------------------

def render_html(content):
    cleaned_html = "\n".join(
        line.strip()
        for line in dedent(content).splitlines()
        if line.strip()
    )

    st.html(cleaned_html)

# --------------------------------------------------
# Styling
# --------------------------------------------------

render_html(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 85% 0%,
                rgba(0, 180, 216, 0.12),
                transparent 30%
            ),
            radial-gradient(
                circle at 10% 90%,
                rgba(0, 119, 182, 0.10),
                transparent 35%
            ),
            #07111f;
        color: #ffffff;
    }

    .block-container {
        max-width: 1480px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #ffffff;
    }

    [data-testid="stSidebar"] {
        background: #0a1422;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #e8eef7;
    }

    .hero {
        padding: 34px;
        border-radius: 24px;
        background:
            linear-gradient(
                135deg,
                rgba(0, 180, 216, 0.18),
                rgba(0, 119, 182, 0.06)
            );
        border: 1px solid rgba(72, 202, 228, 0.28);
        margin-bottom: 24px;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 13px;
        border-radius: 999px;
        background: rgba(72, 202, 228, 0.12);
        border: 1px solid rgba(72, 202, 228, 0.25);
        color: #90e0ef;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 16px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        line-height: 1.12;
        color: #ffffff;
        margin-bottom: 12px;
    }

    .hero-subtitle {
        color: #afbdd0;
        font-size: 17px;
        line-height: 1.7;
        max-width: 860px;
    }

    .status-card {
        padding: 18px 20px;
        border-radius: 18px;
        background: rgba(13, 27, 42, 0.84);
        border: 1px solid rgba(255,255,255,0.08);
        min-height: 96px;
    }

    .metric-card {
        padding: 22px;
        border-radius: 20px;
        background: rgba(13, 27, 42, 0.88);
        border: 1px solid rgba(255,255,255,0.08);
        min-height: 155px;
    }

    .metric-label {
        color: #91a4b9;
        font-size: 14px;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 31px;
        font-weight: 800;
        line-height: 1.2;
    }

    .metric-note {
        color: #8fddec;
        font-size: 13px;
        margin-top: 10px;
    }

    .section-card {
        padding: 24px;
        border-radius: 20px;
        background: rgba(13, 27, 42, 0.84);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 18px;
    }

    .small-muted {
        color: #93a6bc;
        font-size: 13px;
        line-height: 1.6;
    }

    .status-dot {
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #2ecc71;
        margin-right: 7px;
    }

    .weather-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 16px;
    }

    .weather-item {
        padding: 14px;
        border-radius: 14px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.06);
    }

    .weather-label {
        color: #91a4b9;
        font-size: 12px;
        margin-bottom: 5px;
    }

    .weather-value {
        color: #ffffff;
        font-size: 19px;
        font-weight: 700;
    }

    .pipeline-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        justify-content: center;
    }

    .pipeline-node {
        padding: 10px 14px;
        border-radius: 12px;
        background: rgba(72, 202, 228, 0.08);
        border: 1px solid rgba(72, 202, 228, 0.18);
        font-weight: 600;
        color: #e7f7fb;
    }

    .pipeline-arrow {
        color: #6fbfd4;
        font-weight: 700;
    }

    .footer-text {
        text-align: center;
        color: #73869b;
        font-size: 13px;
        padding-top: 30px;
    }

    </style>
    """
)


# --------------------------------------------------
# Config
# --------------------------------------------------

API_BASE_URL = "http://127.0.0.1:8000"


# --------------------------------------------------
# API Helpers
# --------------------------------------------------

@st.cache_data(ttl=30)
def get_api_data(endpoint, params=None):
    response = requests.get(
        f"{API_BASE_URL}{endpoint}",
        params=params,
        timeout=15,
    )

    response.raise_for_status()
    return response.json()


def safe_api_call(endpoint, params=None):
    try:
        return get_api_data(
            endpoint,
            params,
        )

    except requests.exceptions.RequestException as error:
        st.error(
            "FastAPI backend-এর সাথে connect করা যাচ্ছে না.\n\n"
            f"{error}"
        )
        st.stop()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.markdown("## ⚡ BD Power Forecast")

    st.caption(
        "Production-oriented electricity "
        "demand forecasting platform."
    )

    st.divider()

    st.markdown("### Navigation")
    st.markdown("📊 Dashboard")
    st.markdown("📈 Forecast Analytics")
    st.markdown("🧠 Model Performance")
    st.markdown("⚙️ System Status")

    st.divider()

    st.markdown("### Model")
    st.markdown("**Ridge Regression**")
    st.code("alpha = 0.01")
    st.caption("Registered in MLflow Model Registry")

    st.divider()

    refresh = st.button(
        "🔄 Refresh Data",
        use_container_width=True,
    )

    if refresh:
        st.cache_data.clear()
        st.rerun()


# --------------------------------------------------
# Load Data
# --------------------------------------------------

health = safe_api_call("/health")
prediction = safe_api_call("/predict/latest")
model_info = safe_api_call("/model/info")
history = safe_api_call(
    "/history",
    params={"limit": 90},
)


# --------------------------------------------------
# Hero
# --------------------------------------------------

render_html(
    """
    <div class="hero">
        <div class="hero-badge">
            AI-Powered Energy Forecasting
        </div>

        <div class="hero-title">
            Bangladesh Electricity Demand
            Forecasting Platform
        </div>

        <div class="hero-subtitle">
            A production-oriented MLOps platform for
            forecasting next-day national electricity
            demand using BPDB demand data, calendar signals,
            weather features, DVC, MLflow Model Registry,
            FastAPI, and Streamlit.
        </div>
    </div>
    """
)


# --------------------------------------------------
# Status
# --------------------------------------------------

status_col1, status_col2, status_col3 = st.columns(
    [1, 1, 2]
)

with status_col1:
    render_html(
        f"""
        <div class="status-card">
            <span class="status-dot"></span>
            <b>API Status</b>
            <br>
            <span class="small-muted">
                {health["status"].title()}
            </span>
        </div>
        """
    )

with status_col2:
    render_html(
        f"""
        <div class="status-card">
            <span class="status-dot"></span>
            <b>Model Status</b>
            <br>
            <span class="small-muted">
                {health["model_status"].replace("_", " ").title()}
            </span>
        </div>
        """
    )

with status_col3:
    render_html(
        f"""
        <div class="status-card">
            <b>MLflow Model Registry</b>
            <br>
            <span class="small-muted">
                {prediction["model"]["name"]}
                · alias:
                <b>{prediction["model"]["alias"]}</b>
            </span>
        </div>
        """
    )


# --------------------------------------------------
# Forecast Cards
# --------------------------------------------------

st.markdown("## Next-Day Forecast")

metric_col1, metric_col2, metric_col3, metric_col4 = (
    st.columns(4)
)

with metric_col1:
    render_html(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Current Demand
            </div>

            <div class="metric-value">
                {prediction["current_demand_mw"]:,.0f}
                <span style="font-size:16px;">MW</span>
            </div>

            <div class="metric-note">
                Observation:
                {prediction["observation_date"]}
            </div>
        </div>
        """
    )

with metric_col2:
    render_html(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Predicted Demand
            </div>

            <div class="metric-value">
                {prediction["predicted_demand_mw"]:,.0f}
                <span style="font-size:16px;">MW</span>
            </div>

            <div class="metric-note">
                Forecast:
                {prediction["forecast_date"]}
            </div>
        </div>
        """
    )

with metric_col3:
    change = prediction["change_mw"]

    render_html(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Expected Change
            </div>

            <div class="metric-value">
                {change:+,.0f}
                <span style="font-size:16px;">MW</span>
            </div>

            <div class="metric-note">
                {prediction["change_percent"]:+.2f}%
                vs current demand
            </div>
        </div>
        """
    )

with metric_col4:
    render_html(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Holdout MAPE
            </div>

            <div class="metric-value">
                {model_info["holdout_mape"]:.2f}%
            </div>

            <div class="metric-note">
                CV Mean MAPE:
                {model_info["cv_mean_mape"]:.2f}%
            </div>
        </div>
        """
    )


# --------------------------------------------------
# Forecast Overview
# --------------------------------------------------

st.markdown("## Forecast Overview")

forecast_left, forecast_right = st.columns(
    [1.55, 1]
)

with forecast_left:

    fig_compare = go.Figure()

    fig_compare.add_trace(
        go.Bar(
            x=[
                "Current Demand",
                "Next-Day Forecast",
            ],
            y=[
                prediction["current_demand_mw"],
                prediction["predicted_demand_mw"],
            ],
            text=[
                f'{prediction["current_demand_mw"]:,.0f} MW',
                f'{prediction["predicted_demand_mw"]:,.0f} MW',
            ],
            textposition="auto",
        )
    )

    fig_compare.update_layout(
        title="Current vs Predicted National Demand",
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce6f2"),
        yaxis_title="Demand (MW)",
        showlegend=False,
        margin=dict(
            l=30,
            r=30,
            t=70,
            b=30,
        ),
    )

    st.plotly_chart(
        fig_compare,
        use_container_width=True,
    )


with forecast_right:

    weather = prediction["weather"]

    render_html(
        f"""
        <div class="section-card">
            <h3>🌦 Weather Context</h3>

            <div class="small-muted">
                Current-day Dhaka weather used as an
                external forecasting signal.
            </div>

            <div class="weather-grid">

                <div class="weather-item">
                    <div class="weather-label">
                        Mean Temperature
                    </div>
                    <div class="weather-value">
                        {weather["temperature_mean_c"]:.1f} °C
                    </div>
                </div>

                <div class="weather-item">
                    <div class="weather-label">
                        Maximum Temperature
                    </div>
                    <div class="weather-value">
                        {weather["temperature_max_c"]:.1f} °C
                    </div>
                </div>

                <div class="weather-item">
                    <div class="weather-label">
                        Minimum Temperature
                    </div>
                    <div class="weather-value">
                        {weather["temperature_min_c"]:.1f} °C
                    </div>
                </div>

                <div class="weather-item">
                    <div class="weather-label">
                        Rainfall
                    </div>
                    <div class="weather-value">
                        {weather["rain_mm"]:.1f} mm
                    </div>
                </div>

            </div>
        </div>
        """
    )


# --------------------------------------------------
# Historical Analytics
# --------------------------------------------------

st.markdown("## Historical Demand Analytics")

history_df = pd.DataFrame(
    history["records"]
)

history_df["date"] = pd.to_datetime(
    history_df["date"]
)

history_df["forecast_date"] = pd.to_datetime(
    history_df["forecast_date"]
)


fig_history = go.Figure()

fig_history.add_trace(
    go.Scatter(
        x=history_df["date"],
        y=history_df["total_demand_mw"],
        mode="lines+markers",
        name="Current Demand",
    )
)

fig_history.add_trace(
    go.Scatter(
        x=history_df["forecast_date"],
        y=history_df["next_day_actual_demand_mw"],
        mode="lines",
        name="Next-Day Actual Demand",
    )
)

fig_history.update_layout(
    title="National Electricity Demand Trend",
    height=500,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#dce6f2"),
    xaxis_title="Date",
    yaxis_title="Demand (MW)",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    margin=dict(
        l=30,
        r=30,
        t=80,
        b=30,
    ),
)

st.plotly_chart(
    fig_history,
    use_container_width=True,
)


# --------------------------------------------------
# Temperature Relationship
# --------------------------------------------------

st.markdown("## Demand & Temperature")

fig_weather = px.scatter(
    history_df,
    x="temperature_mean_c",
    y="next_day_actual_demand_mw",
    hover_data=["date"],
    labels={
        "temperature_mean_c":
            "Mean Temperature (°C)",
        "next_day_actual_demand_mw":
            "Next-Day Demand (MW)",
    },
    title="Temperature vs Next-Day Electricity Demand",
)

fig_weather.update_layout(
    height=440,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#dce6f2"),
)

st.plotly_chart(
    fig_weather,
    use_container_width=True,
)


# --------------------------------------------------
# Model Performance
# --------------------------------------------------

st.markdown("## Model Performance")

performance_col1, performance_col2 = st.columns(2)

with performance_col1:

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Cross-Validation MAPE",
                "Holdout MAPE",
                "Holdout MAE",
                "Holdout RMSE",
            ],
            "Value": [
                f'{model_info["cv_mean_mape"]:.2f}%',
                f'{model_info["holdout_mape"]:.2f}%',
                f'{model_info["holdout_mae_mw"]:,.2f} MW',
                f'{model_info["holdout_rmse_mw"]:,.2f} MW',
            ],
        }
    )

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True,
    )


with performance_col2:

    render_html(
        f"""
        <div class="section-card">
            <h3>Model Configuration</h3>

            <b>Algorithm</b>
            <br>
            <span class="small-muted">
                {model_info["model_type"]}
            </span>

            <br><br>

            <b>Regularization α</b>
            <br>
            <span class="small-muted">
                {model_info["alpha"]}
            </span>

            <br><br>

            <b>Feature Count</b>
            <br>
            <span class="small-muted">
                {model_info["feature_count"]}
            </span>

            <br><br>

            <b>Forecast Horizon</b>
            <br>
            <span class="small-muted">
                {model_info["forecast_horizon"]}
            </span>

            <br><br>

            <b>Registry Alias</b>
            <br>
            <span class="small-muted">
                champion
            </span>
        </div>
        """
    )


# --------------------------------------------------
# MLOps Pipeline
# --------------------------------------------------

st.markdown("## MLOps Architecture")

render_html(
    """
    <div class="section-card">
        <div class="pipeline-wrap">

            <div class="pipeline-node">BPDB Data</div>
            <div class="pipeline-arrow">→</div>

            <div class="pipeline-node">DVC</div>
            <div class="pipeline-arrow">→</div>

            <div class="pipeline-node">Feature Engineering</div>
            <div class="pipeline-arrow">→</div>

            <div class="pipeline-node">Time-Series CV</div>
            <div class="pipeline-arrow">→</div>

            <div class="pipeline-node">MLflow Tracking</div>
            <div class="pipeline-arrow">→</div>

            <div class="pipeline-node">Model Registry</div>
            <div class="pipeline-arrow">→</div>

            <div class="pipeline-node">FastAPI</div>
            <div class="pipeline-arrow">→</div>

            <div class="pipeline-node">Streamlit UI</div>

        </div>
    </div>
    """
)


# --------------------------------------------------
# Footer
# --------------------------------------------------

render_html(
    """
    <div class="footer-text">
        Bangladesh Electricity Demand Forecasting
        MLOps Platform
        <br>
        Python · DVC · MLflow · FastAPI · Streamlit
    </div>
    """
)