import os
from datetime import date, datetime
from textwrap import dedent

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Bangladesh Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)

REQUEST_TIMEOUT = 30


# ============================================================
# Helper Functions
# ============================================================

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
            detail = response.json().get(
                "detail",
                "Unknown API error",
            )

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


def format_mw(value):
    if value is None:
        return "N/A"

    return f"{format_number(value)} MW"


def format_pct(value):
    if value is None:
        return "N/A"

    return f"{format_number(value)}%"


def scope_description(scope):
    if scope == "holdout":
        return (
            "This date belongs to the 2026 holdout period "
            "and was not used to train the final model."
        )

    if scope == "retrospective_training_period":
        return (
            "This date is inside the final model's training "
            "period. The result is a retrospective model "
            "estimate, not an unseen historical forecast."
        )

    return ""


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 20% 20%,
                rgba(51, 65, 85, 0.18),
                transparent 35%
            ),
            radial-gradient(
                circle at 80% 0%,
                rgba(37, 99, 235, 0.10),
                transparent 30%
            ),
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

    .status-good,
    .status-info,
    .status-warning {
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


# ============================================================
# Backend Health
# ============================================================

health_data, health_error = api_get(
    "/health"
)

api_online = bool(
    health_data
    and health_data.get("status") == "healthy"
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## ⚡ BD Power Forecast")

    st.caption(
        "National and regional electricity demand forecasting."
    )

    st.divider()

    if api_online:
        st.success(
            "FastAPI backend connected"
        )

    else:
        st.error(
            "FastAPI backend offline"
        )

    regions_data, regions_error = api_get(
        "/regions"
    )

    if regions_data:
        region_options = [
            item["region"]
            for item in regions_data["regions"]
        ]

    else:
        region_options = [
            "National",
            "Barisal",
            "Chittagong",
            "Comilla",
            "Dhaka",
            "Khulna",
            "Mymensingh",
            "Rajshahi",
            "Rangpur",
            "Sylhet",
        ]

    selected_region = st.selectbox(
        "Forecast Region",
        region_options,
        index=0,
    )

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

    st.code(
        API_BASE_URL,
        language=None,
    )


# ============================================================
# Header
# ============================================================

render_html(
    """
    <div class="main-header">
        <div class="main-title">
            Bangladesh Electricity Demand Forecasting
        </div>

        <div class="main-subtitle">
            National and regional electricity demand forecasting
            using real BPDB demand data.
        </div>
    </div>
    """
)


if not api_online:
    st.error(
        "FastAPI backend is not available."
    )

    st.code(
        "uvicorn src.api.app:app --reload",
        language="powershell",
    )

    st.stop()


# ============================================================
# Latest Anchor Forecast
# ============================================================

latest_data, latest_error = api_get(
    "/predict/latest",
    params={
        "region": selected_region
    },
)


if latest_data is None:
    st.error(
        f"Could not load latest prediction: {latest_error}"
    )

    st.stop()


st.subheader(
    f"{selected_region} Forecast Overview"
)


latest_change = float(
    latest_data.get(
        "change_from_previous_day_mw",
        0,
    )
    or 0
)

latest_change_percent = float(
    latest_data.get(
        "change_percent",
        0,
    )
    or 0
)

latest_direction_symbol = (
    "+"
    if latest_change >= 0
    else "-"
)


c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "Anchored Forecast",
    format_mw(
        latest_data.get(
            "predicted_demand_mw"
        )
    ),
    help=(
        "Direct next-day forecast based on "
        "the latest real BPDB observation."
    ),
)


c2.metric(
    "Latest Real Demand",
    format_mw(
        latest_data.get(
            "current_demand_mw"
        )
    ),
    help=(
        "Latest real BPDB observation: "
        + str(
            latest_data.get(
                "observation_date",
                "N/A",
            )
        )
    ),
)


c3.metric(
    "Expected Change",
    (
        latest_direction_symbol
        + format_mw(
            abs(latest_change)
        )
    ),
    delta=(
        latest_direction_symbol
        + format_pct(
            abs(latest_change_percent)
        )
    ),
    help=(
        "Expected change relative to "
        "the latest real demand."
    ),
)


c4.metric(
    "Anchor Forecast Date",
    latest_data.get(
        "forecast_date",
        "N/A",
    ),
)


st.info(
    "Latest real BPDB observation: "
    + str(
        latest_data.get(
            "observation_date",
            "N/A",
        )
    )
    + " · Anchored forecast date: "
    + str(
        latest_data.get(
            "forecast_date",
            "N/A",
        )
    )
    + " · Future actual demand is not available yet."
)


# ============================================================
# Latest Forecast Details
# ============================================================

st.markdown(
    "### Latest Anchored Forecast"
)


left, right = st.columns(
    [1.35, 1]
)


with left:
    change = float(
        latest_data.get(
            "change_from_previous_day_mw",
            0,
        )
        or 0
    )

    change_percent = float(
        latest_data.get(
            "change_percent",
            0,
        )
        or 0
    )

    direction = (
        "Increase"
        if change >= 0
        else "Decrease"
    )

    observation_date_text = str(
        latest_data.get(
            "observation_date",
            "N/A",
        )
    )

    forecast_date_text = str(
        latest_data.get(
            "forecast_date",
            "N/A",
        )
    )

    current_demand_text = format_mw(
        latest_data.get(
            "current_demand_mw"
        )
    )

    change_text = format_mw(
        abs(change)
    )

    change_percent_text = format_pct(
        abs(change_percent)
    )

    card_html = f"""
    <div class="section-card">

        <div
            style="
                display:flex;
                justify-content:space-between;
                gap:12px;
                align-items:center;
                margin-bottom:16px;
            "
        >
            <div>

                <div
                    style="
                        color:#94a3b8;
                        font-size:13px;
                    "
                >
                    Forecast Date
                </div>

                <div
                    style="
                        color:#f8fafc;
                        font-size:22px;
                        font-weight:700;
                    "
                >
                    {forecast_date_text}
                </div>

            </div>

            <span class="status-info">
                {selected_region}
            </span>

        </div>

        <div
            style="
                color:#cbd5e1;
                margin-bottom:8px;
            "
        >
            Latest real observation:
            <b>{observation_date_text}</b>
        </div>

        <div
            style="
                color:#cbd5e1;
                margin-bottom:8px;
            "
        >
            Latest real demand:
            <b>{current_demand_text}</b>
        </div>

        <div
            style="
                color:#cbd5e1;
                margin-bottom:8px;
            "
        >
            Expected change:
            <b>
                {direction}
                {change_text}
                ({change_percent_text})
            </b>
        </div>

    </div>
    """

    render_html(card_html)


with right:
    model_info = latest_data.get(
        "model",
        {},
    )

    strategy = model_info.get(
        "strategy",
        "unknown",
    )

    badge_class = (
        "status-good"
        if strategy == "ridge"
        else "status-warning"
    )

    model_name = (
        model_info.get("name")
        or "Persistence baseline"
    )

    alias_name = (
        model_info.get("alias")
        or "N/A"
    )

    model_card_html = f"""
    <div class="section-card">

        <div
            style="
                font-size:17px;
                font-weight:700;
                color:#f8fafc;
                margin-bottom:14px;
            "
        >
            Anchor Strategy
        </div>

        <span class="{badge_class}">
            {strategy.upper()}
        </span>

        <div
            style="
                margin-top:16px;
                color:#94a3b8;
                font-size:12px;
                word-break:break-word;
            "
        >
            {model_name}
        </div>

        <div
            style="
                margin-top:7px;
                color:#64748b;
                font-size:12px;
            "
        >
            Alias: {alias_name}
        </div>

    </div>
    """

    render_html(
        model_card_html
    )


# ============================================================
# Extended Forecast
# ============================================================

st.divider()

st.subheader(
    "Extended Demand Forecast"
)

st.caption(
    "Day+1 uses the Anchored Forecast. "
    "Day+2 through Day+8 use recursive "
    "Bridge Forecasts."
)


extended_data, extended_error = api_get(
    "/predict/extended",
    params={
        "region": selected_region
    },
)


if extended_data:
    forecasts = extended_data.get(
        "forecasts",
        [],
    )

    if forecasts:
        extended_df = pd.DataFrame(
            forecasts
        )

        extended_df[
            "forecast_date"
        ] = pd.to_datetime(
            extended_df[
                "forecast_date"
            ]
        )

        anchor_df = extended_df[
            extended_df[
                "forecast_mode"
            ]
            == "anchor"
        ].copy()

        bridge_df = extended_df[
            extended_df[
                "forecast_mode"
            ]
            == "bridge"
        ].copy()


        latest_real_date = (
            extended_data.get(
                "latest_real_bpdb_date",
                "N/A",
            )
        )

        forecast_end = (
            extended_data.get(
                "forecast_end",
                "N/A",
            )
        )

        validated_horizon = (
            extended_data.get(
                "validated_max_horizon_days",
                8,
            )
        )


        if not anchor_df.empty:
            anchor_prediction = (
                anchor_df[
                    "predicted_demand_mw"
                ]
                .iloc[0]
            )

        else:
            anchor_prediction = None


        e1, e2, e3, e4 = st.columns(
            4
        )


        e1.metric(
            "Latest Real BPDB Date",
            latest_real_date,
        )


        e2.metric(
            "Day+1 Anchor",
            format_mw(
                anchor_prediction
            ),
        )


        e3.metric(
            "Forecast Coverage",
            forecast_end,
        )


        e4.metric(
            "Validated Horizon",
            f"{validated_horizon} Days",
        )


        # ====================================================
        # Extended Forecast Chart
        # ====================================================

        fig_extended = go.Figure()


        if not anchor_df.empty:
            fig_extended.add_trace(
                go.Scatter(
                    x=anchor_df[
                        "forecast_date"
                    ],

                    y=anchor_df[
                        "predicted_demand_mw"
                    ],

                    mode="markers",

                    name="Anchored Forecast",

                    marker=dict(
                        size=12,
                    ),

                    hovertemplate=(
                        "%{x|%d %b %Y}"
                        "<br>"
                        "Anchor: "
                        "%{y:,.2f} MW"
                        "<extra></extra>"
                    ),
                )
            )


        if not bridge_df.empty:
            bridge_plot_df = pd.concat(
                [
                    anchor_df,
                    bridge_df,
                ],
                ignore_index=True,
            )

            fig_extended.add_trace(
                go.Scatter(
                    x=bridge_plot_df[
                        "forecast_date"
                    ],

                    y=bridge_plot_df[
                        "predicted_demand_mw"
                    ],

                    mode="lines+markers",

                    name=(
                        "Extended Bridge Forecast"
                    ),

                    line=dict(
                        width=2.4,
                        dash="dash",
                    ),

                    marker=dict(
                        size=7,
                    ),

                    hovertemplate=(
                        "%{x|%d %b %Y}"
                        "<br>"
                        "Forecast: "
                        "%{y:,.2f} MW"
                        "<extra></extra>"
                    ),
                )
            )


        fig_extended.update_layout(
            height=440,

            margin=dict(
                l=20,
                r=20,
                t=35,
                b=20,
            ),

            paper_bgcolor=(
                "rgba(0,0,0,0)"
            ),

            plot_bgcolor=(
                "rgba(0,0,0,0)"
            ),

            hovermode="x unified",

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),

            xaxis_title=(
                "Forecast Date"
            ),

            yaxis_title=(
                "Demand (MW)"
            ),
        )


        st.plotly_chart(
            fig_extended,
            use_container_width=True,
        )


        # ====================================================
        # Extended Forecast Table
        # ====================================================

        st.markdown(
            "#### Forecast Details"
        )


        table_df = extended_df[
            [
                "forecast_date",
                "horizon_day",
                "display_label",
                "predicted_demand_mw",
                "forecast_status",
                "model_name",
            ]
        ].copy()


        table_df[
            "forecast_date"
        ] = table_df[
            "forecast_date"
        ].dt.strftime(
            "%Y-%m-%d"
        )


        table_df[
            "horizon_day"
        ] = table_df[
            "horizon_day"
        ].apply(
            lambda x: f"Day +{x}"
        )


        table_df[
            "predicted_demand_mw"
        ] = table_df[
            "predicted_demand_mw"
        ].apply(
            lambda x: (
                f"{float(x):,.2f}"
            )
        )


        table_df = table_df.rename(
            columns={
                "forecast_date":
                    "Forecast Date",

                "horizon_day":
                    "Horizon",

                "display_label":
                    "Forecast Type",

                "predicted_demand_mw":
                    "Demand (MW)",

                "forecast_status":
                    "Status",

                "model_name":
                    "Model",
            }
        )


        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
        )


        st.warning(
            "Extended Bridge Forecasts are recursive estimates. "
            "Day+2 and later forecasts depend partly on earlier "
            "predicted demand rather than newly published BPDB "
            "observations. Forecast uncertainty can therefore "
            "increase as the horizon becomes longer."
        )


        st.caption(
            "When new real BPDB demand data becomes available, "
            "the forecast can be re-anchored from the latest "
            "real observation."
        )


    else:
        st.warning(
            "No extended forecast records were returned."
        )

else:
    st.warning(
        "Extended forecast unavailable: "
        + str(extended_error)
    )


# ============================================================
# Specific Historical Date
# ============================================================

st.divider()

st.subheader(
    "Explore a Specific Forecast Date"
)

st.caption(
    "Choose a historical forecast date "
    "to compare the model estimate with "
    "the recorded electricity demand."
)


latest_forecast_date = datetime.strptime(
    latest_data[
        "forecast_date"
    ],
    "%Y-%m-%d",
).date()


default_date = min(
    date(
        2025,
        10,
        10,
    ),
    latest_forecast_date,
)


selected_date = st.date_input(
    "Forecast date",
    value=default_date,
    min_value=date(
        2020,
        2,
        1,
    ),
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
            "forecast_date":
                selected_date.isoformat(),

            "region":
                selected_region,
        },
    )

    if specific_data:
        st.markdown(
            f"### {selected_region} — {selected_date}"
        )


        predicted_text = format_mw(
            specific_data.get(
                "predicted_demand_mw"
            )
        )

        actual_text = format_mw(
            specific_data.get(
                "actual_demand_mw"
            )
        )

        absolute_error_text = format_mw(
            specific_data.get(
                "absolute_error_mw"
            )
        )

        percentage_error_text = format_pct(
            specific_data.get(
                "percentage_error"
            )
        )


        p1, p2, p3, p4 = st.columns(
            4
        )


        p1.metric(
            "Predicted Demand",
            predicted_text,
        )

        p2.metric(
            "Actual Demand",
            actual_text,
        )

        p3.metric(
            "Absolute Error",
            absolute_error_text,
        )

        p4.metric(
            "Percentage Error",
            percentage_error_text,
        )


        scope = specific_data.get(
            "evaluation_scope",
            "",
        )


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


        st.markdown(
            "#### Forecast Context"
        )


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
                    specific_data.get(
                        "observation_date",
                        "N/A",
                    ),

                    specific_data.get(
                        "forecast_date",
                        "N/A",
                    ),

                    format_mw(
                        specific_data.get(
                            "current_demand_mw"
                        )
                    ),

                    predicted_text,

                    actual_text,

                    format_mw(
                        specific_data.get(
                            "forecast_error_mw"
                        )
                    ),
                ],
            }
        )


        st.dataframe(
            context_df,
            use_container_width=True,
            hide_index=True,
        )


    else:
        if isinstance(
            specific_error,
            dict,
        ):
            message = specific_error.get(
                "message",
                "Prediction unavailable",
            )

            st.error(
                message
            )

            start = specific_error.get(
                "available_start"
            )

            end = specific_error.get(
                "available_end"
            )

            if start and end:
                st.info(
                    "Available historical forecast range: "
                    f"{start} to {end}. "
                    "Some dates may be unavailable due "
                    "to source-data filtering."
                )

        else:
            st.error(
                str(specific_error)
            )


# ============================================================
# Historical Chart
# ============================================================

st.divider()

st.subheader(
    "Actual vs Model Estimate"
)

st.caption(
    "For 2026 holdout dates these are unseen "
    "evaluation predictions. Dates inside the "
    "training period are retrospective "
    "final-model estimates."
)


history_data, history_error = api_get(
    "/history",
    params={
        "region":
            selected_region,

        "limit":
            history_days,
    },
)


if history_data:
    history_df = pd.DataFrame(
        history_data.get(
            "records",
            [],
        )
    )

    if history_df.empty:
        st.warning(
            "No historical records returned by the API."
        )

    else:
        history_df["date"] = pd.to_datetime(
            history_df["date"]
        )

        history_df = history_df.sort_values(
            "date"
        )


        fig = go.Figure()


        fig.add_trace(
            go.Scatter(
                x=history_df[
                    "date"
                ],

                y=history_df[
                    "actual_demand_mw"
                ],

                mode="lines",

                name="Actual Demand",

                line=dict(
                    width=2.2
                ),

                hovertemplate=(
                    "%{x|%d %b %Y}"
                    "<br>"
                    "Actual: "
                    "%{y:,.2f} MW"
                    "<extra></extra>"
                ),
            )
        )


        fig.add_trace(
            go.Scatter(
                x=history_df[
                    "date"
                ],

                y=history_df[
                    "predicted_demand_mw"
                ],

                mode="lines",

                name="Model Estimate",

                line=dict(
                    width=2.2
                ),

                hovertemplate=(
                    "%{x|%d %b %Y}"
                    "<br>"
                    "Estimate: "
                    "%{y:,.2f} MW"
                    "<extra></extra>"
                ),
            )
        )


        fig.update_layout(
            height=460,

            margin=dict(
                l=20,
                r=20,
                t=35,
                b=20,
            ),

            paper_bgcolor=(
                "rgba(0,0,0,0)"
            ),

            plot_bgcolor=(
                "rgba(0,0,0,0)"
            ),

            hovermode="x unified",

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),

            xaxis_title=(
                "Forecast Date"
            ),

            yaxis_title=(
                "Demand (MW)"
            ),
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
        )


        avg_mae = history_df[
            "absolute_error_mw"
        ].mean()

        avg_mape = history_df[
            "percentage_error"
        ].mean()

        max_error = history_df[
            "absolute_error_mw"
        ].max()


        h1, h2, h3 = st.columns(
            3
        )


        h1.metric(
            "Window Mean Absolute Error",
            format_mw(avg_mae),
        )


        h2.metric(
            "Window Mean Percentage Error",
            format_pct(avg_mape),
        )


        h3.metric(
            "Largest Absolute Error",
            format_mw(max_error),
        )


else:
    st.warning(
        "Historical data unavailable: "
        + str(history_error)
    )


# ============================================================
# Forecast Model Information
# ============================================================

st.divider()

st.subheader(
    "Forecast Model Information"
)


model_data, model_error = api_get(
    "/model/info",
    params={
        "region":
            selected_region
    },
)


if model_data:
    anchor_info = model_data.get(
        "anchor",
        {},
    )

    bridge_info = model_data.get(
        "bridge",
        {},
    )


    model_col1, model_col2 = st.columns(
        2
    )


    with model_col1:
        anchor_strategy = (
            anchor_info.get(
                "strategy",
                "N/A",
            )
        )

        anchor_model_name = (
            anchor_info.get(
                "model_name"
            )
            or "Persistence baseline"
        )

        anchor_alias = (
            anchor_info.get(
                "alias"
            )
            or "N/A"
        )

        anchor_badge = (
            "status-good"
            if anchor_strategy == "ridge"
            else "status-warning"
        )


        anchor_html = f"""
        <div class="section-card">

            <div
                style="
                    color:#94a3b8;
                    font-size:12px;
                    margin-bottom:6px;
                "
            >
                DAY +1 ANCHOR MODEL
            </div>

            <span class="{anchor_badge}">
                {anchor_strategy.upper()}
            </span>

            <div
                style="
                    color:#f8fafc;
                    font-size:15px;
                    font-weight:650;
                    margin-top:15px;
                    word-break:break-word;
                "
            >
                {anchor_model_name}
            </div>

            <div
                style="
                    color:#94a3b8;
                    font-size:12px;
                    margin-top:8px;
                "
            >
                Alias: {anchor_alias}
            </div>

        </div>
        """

        render_html(
            anchor_html
        )


    with model_col2:
        bridge_enabled = bridge_info.get(
            "enabled",
            False,
        )

        bridge_model_name = bridge_info.get(
            "model_name",
            "N/A",
        )

        bridge_alias = (
            bridge_info.get(
                "alias"
            )
            or "N/A"
        )

        bridge_horizon = bridge_info.get(
            "validated_max_horizon_days",
            "N/A",
        )

        enabled_text = (
            "ENABLED"
            if bridge_enabled
            else "DISABLED"
        )


        bridge_html = f"""
        <div class="section-card">

            <div
                style="
                    color:#94a3b8;
                    font-size:12px;
                    margin-bottom:6px;
                "
            >
                DAY +2 TO DAY +8 BRIDGE MODEL
            </div>

            <span class="status-info">
                {enabled_text}
            </span>

            <div
                style="
                    color:#f8fafc;
                    font-size:15px;
                    font-weight:650;
                    margin-top:15px;
                    word-break:break-word;
                "
            >
                {bridge_model_name}
            </div>

            <div
                style="
                    color:#94a3b8;
                    font-size:12px;
                    margin-top:8px;
                "
            >
                Alias: {bridge_alias}
                · Validated horizon:
                {bridge_horizon} days
            </div>

        </div>
        """

        render_html(
            bridge_html
        )


    if selected_region == "National":
        national_cv_mape = (
            anchor_info.get(
                "cv_mean_mape"
            )
        )

        national_holdout_mape = (
            anchor_info.get(
                "holdout_mape"
            )
        )

        bridge_backtest_mape = (
            bridge_info.get(
                "backtest_mape"
            )
        )


        m1, m2, m3 = st.columns(
            3
        )


        m1.metric(
            "Anchor CV MAPE",
            format_pct(
                national_cv_mape
            ),
        )


        m2.metric(
            "2026 Holdout MAPE",
            format_pct(
                national_holdout_mape
            ),
        )


        m3.metric(
            "Bridge Backtest MAPE",
            format_pct(
                bridge_backtest_mape
            ),
        )


else:
    st.warning(
        "Model information unavailable: "
        + str(model_error)
    )


# ============================================================
# Data Availability Note
# ============================================================

st.divider()

st.info(
    "Data availability note: forecasts are anchored "
    "to the latest real BPDB observation. If BPDB has "
    "not yet published recent area-wise demand data, "
    "Bridge Forecasts extend demand estimates within "
    "the validated 8-day horizon. Imputed demand values "
    "are not treated as real live observations."
)


# ============================================================
# Footer
# ============================================================

render_html(
    """
    <div class="footer-text">
        © 2026 Md. Estiak Rahman Ayon.
        All rights reserved.
        <br>
        Bangladesh Electricity Demand Forecasting
        · FastAPI · MLflow · DVC · Streamlit
    </div>
    """
)