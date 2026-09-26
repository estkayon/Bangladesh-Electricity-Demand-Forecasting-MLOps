import os
from datetime import date, datetime
from textwrap import dedent

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


st.set_page_config(
    page_title="Bangladesh Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 30


def render_html(content):
    cleaned_html = "\n".join(
        line.strip()
        for line in dedent(content).splitlines()
        if line.strip()
    )
    st.html(cleaned_html)


def api_get(endpoint, params=None):
    try:
        response = requests.get(
            f"{API_BASE_URL}{endpoint}",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 200:
            return response.json(), None

        try:
            detail = response.json().get("detail", "Unknown API error")
        except Exception:
            detail = response.text

        return None, detail

    except requests.exceptions.ConnectionError:
        return None, (
            "Could not connect to the FastAPI backend. "
            "Make sure the API is running."
        )
    except requests.exceptions.Timeout:
        return None, "API request timed out."
    except Exception as error:
        return None, str(error)


def format_number(value, decimals=2):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return str(value)


def scope_description(scope):
    if scope == "holdout":
        return (
            "This date belongs to the 2026 holdout period and was not used "
            "to train the final model."
        )
    if scope == "retrospective_training_period":
        return (
            "This date is inside the final model's training period. "
            "The result is a retrospective model estimate, not an unseen "
            "historical forecast."
        )
    return ""


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 20% 20%, rgba(51, 65, 85, 0.18), transparent 35%),
            radial-gradient(circle at 80% 0%, rgba(37, 99, 235, 0.10), transparent 30%),
            #080d18;
    }
    [data-testid="stSidebar"] {
        background: #0c1322;
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }
    .main-header {
        padding: 22px 26px;
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: rgba(15, 23, 42, 0.72);
        margin-bottom: 22px;
    }
    .main-title {
        font-size: 32px;
        font-weight: 750;
        margin: 0;
        color: #f8fafc;
    }
    .main-subtitle {
        margin-top: 8px;
        color: #94a3b8;
        font-size: 15px;
    }
    .section-card {
        padding: 20px;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: rgba(15, 23, 42, 0.72);
        margin-bottom: 16px;
    }
    .status-good, .status-info, .status-warning {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
    }
    .status-good {
        background: rgba(34, 197, 94, 0.13);
        border: 1px solid rgba(34, 197, 94, 0.28);
        color: #86efac;
    }
    .status-info {
        background: rgba(59, 130, 246, 0.13);
        border: 1px solid rgba(59, 130, 246, 0.28);
        color: #93c5fd;
    }
    .status-warning {
        background: rgba(245, 158, 11, 0.13);
        border: 1px solid rgba(245, 158, 11, 0.28);
        color: #fcd34d;
    }
    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        padding-top: 35px;
        padding-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

health_data, health_error = api_get("/health")
api_online = bool(health_data and health_data.get("status") == "healthy")

with st.sidebar:
    st.markdown("## ⚡ BD Power Forecast")
    st.caption("National and regional next-day electricity demand forecasting.")
    st.divider()

    if api_online:
        st.success("FastAPI backend connected")
    else:
        st.error("FastAPI backend offline")

    regions_data, regions_error = api_get("/regions")

    if regions_data:
        region_options = [item["region"] for item in regions_data["regions"]]
    else:
        region_options = [
            "National", "Barisal", "Chittagong", "Comilla", "Dhaka",
            "Khulna", "Mymensingh", "Rajshahi", "Rangpur", "Sylhet",
        ]

    selected_region = st.selectbox("Forecast Region", region_options, index=0)

    st.divider()
    history_days = st.slider(
        "Historical chart length",
        min_value=30,
        max_value=365,
        value=120,
        step=30,
    )

    st.divider()
    st.caption("Backend")
    st.code(API_BASE_URL, language=None)


render_html(
    """
    <div class="main-header">
        <div class="main-title">Bangladesh Electricity Demand Forecasting</div>
        <div class="main-subtitle">
            Production-oriented MLOps platform for national and regional
            next-day electricity demand forecasting.
        </div>
    </div>
    """
)

if not api_online:
    st.error("FastAPI backend is not available.")
    st.code("uvicorn src.api.app:app --reload", language="powershell")
    st.stop()


latest_data, latest_error = api_get(
    "/predict/latest",
    params={"region": selected_region},
)

if latest_data is None:
    st.error(f"Could not load latest prediction: {latest_error}")
    st.stop()


st.subheader(f"{selected_region} Forecast Overview")

c1, c2, c3, c4 = st.columns(4)

latest_change = float(latest_data.get("change_from_previous_day_mw", 0) or 0)
latest_change_percent = float(latest_data.get("change_percent", 0) or 0)
latest_direction_symbol = "+" if latest_change >= 0 else "-"

c1.metric(
    "Predicted Demand",
    f"{format_number(latest_data.get('predicted_demand_mw'))} MW",
    help="Next-day demand forecast based on the latest real BPDB observation.",
)
c2.metric(
    "Current Demand",
    f"{format_number(latest_data.get('current_demand_mw'))} MW",
    help=f"Latest real BPDB observation: {latest_data.get('observation_date', 'N/A')}",
)
c3.metric(
    "Expected Change",
    f"{latest_direction_symbol}{format_number(abs(latest_change))} MW",
    delta=f"{latest_direction_symbol}{format_number(abs(latest_change_percent))}%",
    help="Predicted change relative to the latest observed demand.",
)
c4.metric(
    "Forecast Date",
    latest_data.get("forecast_date", "N/A"),
    help="Actual demand is not yet available for this live forecast.",
)

st.info(
    f"Live forecast · Latest real BPDB observation: "
    f"{latest_data.get('observation_date', 'N/A')} · "
    f"Future actual demand for {latest_data.get('forecast_date', 'N/A')} "
    "is not available yet."
)


st.markdown("### Latest Available Forecast")
left, right = st.columns([1.35, 1])

with left:
    change = float(latest_data.get("change_from_previous_day_mw", 0) or 0)
    change_percent = float(latest_data.get("change_percent", 0) or 0)
    direction = "Increase" if change >= 0 else "Decrease"

    observation_date_text = latest_data.get("observation_date", "N/A")
    forecast_date_text = latest_data.get("forecast_date", "N/A")
    current_demand_text = format_number(latest_data.get("current_demand_mw"))
    change_text = format_number(abs(change))
    change_percent_text = format_number(abs(change_percent))

    render_html(
        f"""
        <div class="section-card">
            <div style="display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:16px;">
                <div>
                    <div style="color:#94a3b8;font-size:13px;">Forecast Date</div>
                    <div style="color:#f8fafc;font-size:22px;font-weight:700;">{forecast_date_text}</div>
                </div>
                <span class="status-info">{selected_region}</span>
            </div>
            <div style="color:#cbd5e1;margin-bottom:8px;">Observation date: <b>{observation_date_text}</b></div>
            <div style="color:#cbd5e1;margin-bottom:8px;">Previous demand: <b>{current_demand_text} MW</b></div>
            <div style="color:#cbd5e1;margin-bottom:8px;">Expected change: <b>{direction} {change_text} MW ({change_percent_text}%)</b></div>
        </div>
        """
    )

with right:
    model_info = latest_data.get("model", {})
    strategy = model_info.get("strategy", "unknown")
    badge_class = "status-good" if strategy == "ridge" else "status-warning"
    model_source = model_info.get("source", "N/A")
    model_name = model_info.get("name") or "Persistence baseline"

    render_html(
        f"""
        <div class="section-card">
            <div style="font-size:17px;font-weight:700;color:#f8fafc;margin-bottom:14px;">Deployment Strategy</div>
            <span class="{badge_class}">{strategy.upper()}</span>
            <div style="margin-top:16px;color:#cbd5e1;font-size:13px;">Source: {model_source}</div>
            <div style="margin-top:7px;color:#94a3b8;font-size:12px;word-break:break-word;">{model_name}</div>
        </div>
        """
    )


st.divider()
st.subheader("Explore a Specific Forecast Date")
st.caption(
    "Choose a historical forecast date to compare the model estimate "
    "with the recorded electricity demand."
)

latest_forecast_date = datetime.strptime(
    latest_data["forecast_date"], "%Y-%m-%d"
).date()

default_date = min(date(2025, 10, 10), latest_forecast_date)

selected_date = st.date_input(
    "Forecast date",
    value=default_date,
    min_value=date(2020, 2, 1),
    max_value=latest_forecast_date,
)

if st.button(
    "Generate Date-Specific Prediction",
    type="primary",
    use_container_width=True,
):
    specific_data, specific_error = api_get(
        "/predict/date",
        params={
            "forecast_date": selected_date.isoformat(),
            "region": selected_region,
        },
    )

    if specific_data:
        st.markdown(f"### {selected_region} — {selected_date}")

        predicted_text = f"{format_number(specific_data.get('predicted_demand_mw'))} MW"
        actual_text = f"{format_number(specific_data.get('actual_demand_mw'))} MW"
        absolute_error_text = f"{format_number(specific_data.get('absolute_error_mw'))} MW"
        percentage_error_text = f"{format_number(specific_data.get('percentage_error'))}%"

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Predicted Demand", predicted_text)
        p2.metric("Actual Demand", actual_text)
        p3.metric("Absolute Error", absolute_error_text)
        p4.metric("Percentage Error", percentage_error_text)

        scope = specific_data.get("evaluation_scope", "")

        if scope == "holdout":
            st.success(
                "Evaluation scope: Holdout Evaluation — "
                + scope_description(scope)
            )
        else:
            st.warning(
                "Evaluation scope: Retrospective Estimate — "
                + scope_description(scope)
            )

        detail_left, detail_right = st.columns(2)

        with detail_left:
            st.markdown("#### Forecast Context")

            context_df = pd.DataFrame(
                {
                    "Metric": [
                        "Observation Date",
                        "Forecast Date",
                        "Previous Demand",
                        "Predicted Demand",
                        "Actual Demand",
                        "Forecast Error",
                    ],
                    "Value": [
                        specific_data.get("observation_date", "N/A"),
                        specific_data.get("forecast_date", "N/A"),
                        f"{format_number(specific_data.get('current_demand_mw'))} MW",
                        predicted_text,
                        actual_text,
                        f"{format_number(specific_data.get('forecast_error_mw'))} MW",
                    ],
                }
            )

            st.dataframe(
                context_df,
                use_container_width=True,
                hide_index=True,
            )

        with detail_right:
            st.markdown("#### Weather Context")
            weather = specific_data.get("weather", {})

            weather_df = pd.DataFrame(
                {
                    "Metric": [
                        "Maximum Temperature",
                        "Minimum Temperature",
                        "Mean Temperature",
                        "Precipitation",
                        "Rain",
                    ],
                    "Value": [
                        f"{format_number(weather.get('temperature_max_c'))} °C",
                        f"{format_number(weather.get('temperature_min_c'))} °C",
                        f"{format_number(weather.get('temperature_mean_c'))} °C",
                        f"{format_number(weather.get('precipitation_mm'))} mm",
                        f"{format_number(weather.get('rain_mm'))} mm",
                    ],
                }
            )

            st.dataframe(
                weather_df,
                use_container_width=True,
                hide_index=True,
            )

    else:
        if isinstance(specific_error, dict):
            message = specific_error.get("message", "Prediction unavailable")
            st.error(message)

            start = specific_error.get("available_start")
            end = specific_error.get("available_end")

            if start and end:
                st.info(
                    f"Overall available forecast range: {start} to {end}. "
                    "Some individual dates may be unavailable due to "
                    "source-data quality filtering."
                )
        else:
            st.error(str(specific_error))


st.divider()
st.subheader("Actual vs Model Estimate")
st.caption(
    "For 2026 holdout dates these are unseen evaluation predictions. "
    "Dates inside the training period are retrospective final-model estimates."
)

history_data, history_error = api_get(
    "/history",
    params={
        "region": selected_region,
        "limit": history_days,
    },
)

if history_data:
    history_df = pd.DataFrame(history_data.get("records", []))

    if history_df.empty:
        st.warning("No historical records returned by the API.")
    else:
        history_df["date"] = pd.to_datetime(history_df["date"])
        history_df = history_df.sort_values("date")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=history_df["date"],
                y=history_df["actual_demand_mw"],
                mode="lines",
                name="Actual Demand",
                line=dict(width=2.2),
                hovertemplate=(
                    "%{x|%d %b %Y}<br>Actual: %{y:,.2f} MW<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=history_df["date"],
                y=history_df["predicted_demand_mw"],
                mode="lines",
                name="Model Estimate",
                line=dict(width=2.2),
                hovertemplate=(
                    "%{x|%d %b %Y}<br>Estimate: %{y:,.2f} MW<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            height=460,
            margin=dict(l=20, r=20, t=35, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            xaxis_title="Forecast Date",
            yaxis_title="Demand (MW)",
        )

        st.plotly_chart(fig, use_container_width=True)

        avg_mae = history_df["absolute_error_mw"].mean()
        avg_mape = history_df["percentage_error"].mean()
        max_error = history_df["absolute_error_mw"].max()

        h1, h2, h3 = st.columns(3)
        h1.metric("Window Mean Absolute Error", f"{format_number(avg_mae)} MW")
        h2.metric("Window Mean Percentage Error", f"{format_number(avg_mape)}%")
        h3.metric("Largest Absolute Error", f"{format_number(max_error)} MW")
else:
    st.warning(f"Historical data unavailable: {history_error}")


st.divider()
st.subheader("Model & MLOps Information")

model_data, model_error = api_get(
    "/model/info",
    params={"region": selected_region},
)

if model_data:
    model_col1, model_col2 = st.columns(2)

    with model_col1:
        strategy = model_data.get("strategy", "N/A")
        region_name = model_data.get("region", selected_region)

        render_html(
            f"""
            <div class="section-card">
                <div style="color:#94a3b8;font-size:12px;margin-bottom:6px;">DEPLOYMENT STRATEGY</div>
                <div style="color:#f8fafc;font-size:22px;font-weight:700;">{strategy.upper()}</div>
                <div style="color:#94a3b8;font-size:13px;margin-top:10px;">Region: {region_name}</div>
            </div>
            """
        )

    with model_col2:
        active_model_name = model_data.get("model_name") or "Persistence baseline"
        alias_name = model_data.get("alias") or "N/A"

        render_html(
            f"""
            <div class="section-card">
                <div style="color:#94a3b8;font-size:12px;margin-bottom:6px;">ACTIVE MODEL</div>
                <div style="color:#f8fafc;font-size:16px;font-weight:650;word-break:break-word;">{active_model_name}</div>
                <div style="color:#94a3b8;font-size:13px;margin-top:10px;">Alias: {alias_name}</div>
            </div>
            """
        )
else:
    st.warning(f"Model info unavailable: {model_error}")


st.divider()
st.subheader("MLOps Pipeline")

st.code(
    """
BPDB Electricity Data
        │
        ▼
Data Collection
        │
        ▼
Preprocessing & Missing-Date Handling
        │
        ├── National Dataset
        └── Regional Dataset
        │
        ▼
Feature Engineering
        │
        ├── Training / Evaluation Features
        ├── Live Inference Features
        ├── Calendar Features
        ├── Bangladesh Holidays
        ├── Weather Features
        ├── Exact Lag Features
        └── Rolling Demand Features
        │
        ▼
Time-Series Cross Validation
        │
        ├── Ridge Regression
        └── Persistence Baseline
        │
        ▼
Region-Specific Model Strategy
        │
        ▼
MLflow / DagsHub Model Registry
        │
        ▼
Champion Model Alias
        │
        ▼
FastAPI Prediction Service
        │
        ▼
Streamlit Dashboard
    """,
    language=None,
)

st.info(
    "Data availability note: live next-day forecasts use the latest real BPDB "
    "observation. If BPDB has not published recent area-wise demand data, the "
    "latest forecast date will also lag behind the current calendar date. "
    "Imputed demand values are not used as live observations."
)

render_html(
    """
    <div class="footer-text">
        © 2026 Md. Estiak Rahman Ayon. All rights reserved.
        <br>
        Bangladesh Electricity Demand Forecasting MLOps Platform
        · FastAPI · MLflow · DVC · Streamlit
    </div>
    """
)
