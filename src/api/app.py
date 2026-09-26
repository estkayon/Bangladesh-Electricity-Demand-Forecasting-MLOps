import os
from datetime import date
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.logger import get_logger


logger = get_logger("api")


# ============================================================
# Configuration
# ============================================================

NATIONAL_FEATURE_PATH = (
    "data/processed/model_features.csv"
)

REGIONAL_FEATURE_PATH = (
    "data/processed/regional_model_features.csv"
)

NATIONAL_INFERENCE_PATH = (
    "data/processed/national_inference_features.csv"
)

REGIONAL_INFERENCE_PATH = (
    "data/processed/regional_inference_features.csv"
)

NATIONAL_BRIDGE_FORECAST_PATH = Path(
    "data/processed/bridge_forecast.csv"
)

REGIONAL_BRIDGE_FORECAST_PATH = Path(
    "data/processed/regional_bridge_forecast.csv"
)


NATIONAL_MODEL_NAME = (
    "bangladesh-electricity-demand-ridge"
)

NATIONAL_BRIDGE_MODEL_NAME = (
    "bangladesh-electricity-demand-bridge"
)

MODEL_ALIAS = "champion"

MAX_BRIDGE_HORIZON = 8


REGIONS = [
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


REGIONAL_MODEL_NAMES = {
    "Chittagong":
        "bangladesh-electricity-demand-chittagong-ridge",

    "Comilla":
        "bangladesh-electricity-demand-comilla-ridge",

    "Dhaka":
        "bangladesh-electricity-demand-dhaka-ridge",

    "Khulna":
        "bangladesh-electricity-demand-khulna-ridge",

    "Mymensingh":
        "bangladesh-electricity-demand-mymensingh-ridge",

    "Rangpur":
        "bangladesh-electricity-demand-rangpur-ridge",

    "Sylhet":
        "bangladesh-electricity-demand-sylhet-ridge",
}


REGIONAL_BRIDGE_MODEL_NAMES = {
    "Barisal":
        "bangladesh-electricity-demand-barisal-bridge",

    "Chittagong":
        "bangladesh-electricity-demand-chittagong-bridge",

    "Comilla":
        "bangladesh-electricity-demand-comilla-bridge",

    "Dhaka":
        "bangladesh-electricity-demand-dhaka-bridge",

    "Khulna":
        "bangladesh-electricity-demand-khulna-bridge",

    "Mymensingh":
        "bangladesh-electricity-demand-mymensingh-bridge",

    "Rajshahi":
        "bangladesh-electricity-demand-rajshahi-bridge",

    "Rangpur":
        "bangladesh-electricity-demand-rangpur-bridge",

    "Sylhet":
        "bangladesh-electricity-demand-sylhet-bridge",
}


REGIONAL_STRATEGIES = {
    "Barisal": "baseline",
    "Chittagong": "ridge",
    "Comilla": "ridge",
    "Dhaka": "ridge",
    "Khulna": "ridge",
    "Mymensingh": "ridge",
    "Rajshahi": "baseline",
    "Rangpur": "ridge",
    "Sylhet": "ridge",
}


NATIONAL_FEATURE_COLUMNS = [
    "total_demand",
    "total_load_shed",
    "day_of_week",
    "month",
    "day_of_month",
    "day_of_year",
    "is_weekend",
    "is_holiday",
    "trend_days",
    "temperature_max_c",
    "temperature_min_c",
    "temperature_mean_c",
    "precipitation_mm",
    "rain_mm",
    "lag_1_day",
    "lag_7_day",
    "lag_14_day",
    "rolling_7_day_mean",
    "rolling_14_day_mean",
    "rolling_30_day_mean",
    "load_shed_lag_1_day",
]


REGIONAL_FEATURE_COLUMNS = [
    "regional_demand_mw",
    "regional_load_shed_mw",
    "day_of_week",
    "month",
    "day_of_month",
    "day_of_year",
    "is_weekend",
    "is_holiday",
    "trend_days",
    "temperature_max_c",
    "temperature_min_c",
    "temperature_mean_c",
    "precipitation_mm",
    "rain_mm",
    "lag_1_day",
    "lag_7_day",
    "lag_14_day",
    "rolling_7_day_mean",
    "rolling_14_day_mean",
    "rolling_30_day_mean",
    "load_shed_lag_1_day",
]


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title=(
        "Bangladesh Electricity Demand "
        "Forecasting API"
    ),
    description=(
        "National and regional electricity "
        "demand forecasting with Anchor and "
        "Extended Bridge Forecasts."
    ),
    version="5.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


model_cache = {}


# ============================================================
# MLflow
# ============================================================

def setup_mlflow():
    load_dotenv()

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI"
    )

    if not tracking_uri:
        raise ValueError(
            "MLFLOW_TRACKING_URI not found"
        )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    logger.info(
        "MLflow tracking configured"
    )


def load_registered_model(
    model_name,
):
    if model_name in model_cache:
        return model_cache[
            model_name
        ]

    model_uri = (
        f"models:/{model_name}"
        f"@{MODEL_ALIAS}"
    )

    logger.info(
        f"Loading model: {model_uri}"
    )

    model = mlflow.sklearn.load_model(
        model_uri
    )

    model_cache[
        model_name
    ] = model

    logger.info(
        f"Model loaded successfully: "
        f"{model_name}"
    )

    return model


# ============================================================
# Historical Data
# ============================================================

def load_national_data():
    df = pd.read_csv(
        NATIONAL_FEATURE_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"]
    )

    return (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )


def load_regional_data(
    region=None,
):
    df = pd.read_csv(
        REGIONAL_FEATURE_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"]
    )

    if region:
        df = df[
            df["region"] == region
        ].copy()

    return (
        df
        .sort_values(
            [
                "region",
                "Date",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# Live Inference Data
# ============================================================

def load_national_inference_data():
    df = pd.read_csv(
        NATIONAL_INFERENCE_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"]
    )

    return (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )


def load_regional_inference_data(
    region=None,
):
    df = pd.read_csv(
        REGIONAL_INFERENCE_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"]
    )

    if region:
        df = df[
            df["region"] == region
        ].copy()

    return (
        df
        .sort_values(
            [
                "region",
                "Date",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# Extended Forecast Data
# ============================================================

def load_national_bridge_data():
    if not NATIONAL_BRIDGE_FORECAST_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "National Bridge forecast "
                "file not found."
            ),
        )

    df = pd.read_csv(
        NATIONAL_BRIDGE_FORECAST_PATH
    )

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "National Bridge forecast "
                "dataset is empty."
            ),
        )

    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"]
    )

    df["latest_real_bpdb_date"] = (
        pd.to_datetime(
            df[
                "latest_real_bpdb_date"
            ]
        )
    )

    return (
        df
        .sort_values(
            "forecast_date"
        )
        .reset_index(drop=True)
    )


def load_regional_bridge_data(
    region,
):
    if not REGIONAL_BRIDGE_FORECAST_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Regional Bridge forecast "
                "file not found."
            ),
        )

    df = pd.read_csv(
        REGIONAL_BRIDGE_FORECAST_PATH
    )

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "Regional Bridge forecast "
                "dataset is empty."
            ),
        )

    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"]
    )

    df["latest_real_bpdb_date"] = (
        pd.to_datetime(
            df[
                "latest_real_bpdb_date"
            ]
        )
    )

    df = df[
        df["region"] == region
    ].copy()

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No extended forecast "
                f"available for {region}"
            ),
        )

    return (
        df
        .sort_values(
            "forecast_date"
        )
        .reset_index(drop=True)
    )


# ============================================================
# Validation
# ============================================================

def validate_region(
    region,
):
    if region not in REGIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "message":
                    "Invalid region",

                "available_regions":
                    REGIONS,
            },
        )


def get_evaluation_scope(
    forecast_date,
):
    if (
        forecast_date
        >= pd.Timestamp(
            "2026-01-01"
        )
    ):
        return "holdout"

    return (
        "retrospective_training_period"
    )


# ============================================================
# Prediction Helpers
# ============================================================

def predict_national_row(
    row,
):
    model = load_registered_model(
        NATIONAL_MODEL_NAME
    )

    X = pd.DataFrame(
        [
            row[
                NATIONAL_FEATURE_COLUMNS
            ].to_dict()
        ]
    )

    return float(
        model.predict(X)[0]
    )


def predict_regional_row(
    row,
    region,
):
    strategy = (
        REGIONAL_STRATEGIES[
            region
        ]
    )

    if strategy == "baseline":
        return float(
            row[
                "regional_demand_mw"
            ]
        )

    model_name = (
        REGIONAL_MODEL_NAMES[
            region
        ]
    )

    model = load_registered_model(
        model_name
    )

    X = pd.DataFrame(
        [
            row[
                REGIONAL_FEATURE_COLUMNS
            ].to_dict()
        ]
    )

    return float(
        model.predict(X)[0]
    )


# ============================================================
# Historical Response
# ============================================================

def build_prediction_response(
    row,
    region,
):
    forecast_date = pd.Timestamp(
        row["forecast_date"]
    )

    observation_date = pd.Timestamp(
        row["Date"]
    )

    if region == "National":
        current_demand = float(
            row["total_demand"]
        )

        actual_demand = float(
            row[
                "next_day_total_demand"
            ]
        )

        prediction = (
            predict_national_row(
                row
            )
        )

        strategy = "ridge"

        model_name = (
            NATIONAL_MODEL_NAME
        )

    else:
        current_demand = float(
            row[
                "regional_demand_mw"
            ]
        )

        actual_demand = float(
            row[
                "next_day_demand_mw"
            ]
        )

        prediction = (
            predict_regional_row(
                row,
                region,
            )
        )

        strategy = (
            REGIONAL_STRATEGIES[
                region
            ]
        )

        model_name = (
            REGIONAL_MODEL_NAMES.get(
                region
            )
        )

    change = (
        prediction
        - current_demand
    )

    forecast_error = (
        prediction
        - actual_demand
    )

    absolute_error = abs(
        forecast_error
    )

    if actual_demand != 0:
        percentage_error = (
            absolute_error
            / actual_demand
            * 100
        )
    else:
        percentage_error = 0.0

    if current_demand != 0:
        change_percent = (
            change
            / current_demand
            * 100
        )
    else:
        change_percent = 0.0

    return {
        "region":
            region,

        "prediction_type":
            "historical_evaluation",

        "observation_date":
            str(
                observation_date.date()
            ),

        "forecast_date":
            str(
                forecast_date.date()
            ),

        "current_demand_mw":
            round(
                current_demand,
                2,
            ),

        "predicted_demand_mw":
            round(
                prediction,
                2,
            ),

        "actual_demand_mw":
            round(
                actual_demand,
                2,
            ),

        "change_from_previous_day_mw":
            round(
                change,
                2,
            ),

        "change_percent":
            round(
                change_percent,
                2,
            ),

        "forecast_error_mw":
            round(
                forecast_error,
                2,
            ),

        "absolute_error_mw":
            round(
                absolute_error,
                2,
            ),

        "percentage_error":
            round(
                percentage_error,
                2,
            ),

        "evaluation_scope":
            get_evaluation_scope(
                forecast_date
            ),

        "model": {
            "strategy":
                strategy,

            "name":
                model_name,

            "alias":
                (
                    MODEL_ALIAS
                    if model_name
                    else None
                ),
        },
    }


# ============================================================
# Live Response
# ============================================================

def build_live_prediction_response(
    row,
    region,
):
    forecast_date = pd.Timestamp(
        row["forecast_date"]
    )

    observation_date = pd.Timestamp(
        row["Date"]
    )

    if region == "National":
        current_demand = float(
            row["total_demand"]
        )

        prediction = (
            predict_national_row(
                row
            )
        )

        strategy = "ridge"

        model_name = (
            NATIONAL_MODEL_NAME
        )

    else:
        current_demand = float(
            row[
                "regional_demand_mw"
            ]
        )

        prediction = (
            predict_regional_row(
                row,
                region,
            )
        )

        strategy = (
            REGIONAL_STRATEGIES[
                region
            ]
        )

        model_name = (
            REGIONAL_MODEL_NAMES.get(
                region
            )
        )

    change = (
        prediction
        - current_demand
    )

    if current_demand != 0:
        change_percent = (
            change
            / current_demand
            * 100
        )
    else:
        change_percent = 0.0

    return {
        "region":
            region,

        "prediction_type":
            "live_inference",

        "observation_date":
            str(
                observation_date.date()
            ),

        "forecast_date":
            str(
                forecast_date.date()
            ),

        "current_demand_mw":
            round(
                current_demand,
                2,
            ),

        "predicted_demand_mw":
            round(
                prediction,
                2,
            ),

        "actual_demand_mw":
            None,

        "change_from_previous_day_mw":
            round(
                change,
                2,
            ),

        "change_percent":
            round(
                change_percent,
                2,
            ),

        "forecast_error_mw":
            None,

        "absolute_error_mw":
            None,

        "percentage_error":
            None,

        "evaluation_scope":
            "live_forecast",

        "data_status": {
            "latest_real_observation_date":
                str(
                    observation_date.date()
                ),

            "future_actual_available":
                False,
        },

        "model": {
            "strategy":
                strategy,

            "name":
                model_name,

            "alias":
                (
                    MODEL_ALIAS
                    if model_name
                    else None
                ),
        },
    }


# ============================================================
# Extended Forecast Response
# ============================================================

def build_extended_response(
    df,
    region,
):
    latest_real_date = (
        df[
            "latest_real_bpdb_date"
        ]
        .iloc[0]
    )

    forecasts = []

    for _, row in df.iterrows():
        mode = str(
            row[
                "forecast_mode"
            ]
        )

        horizon_day = int(
            row[
                "horizon_day"
            ]
        )

        if mode == "anchor":
            display_label = (
                "Anchored Forecast"
            )

            confidence_note = (
                "Direct next-day forecast "
                "based on the latest real "
                "BPDB observation."
            )

        else:
            display_label = (
                "Extended Bridge Forecast"
            )

            confidence_note = (
                "Recursive estimate beyond "
                "the latest published BPDB "
                "observation."
            )

        actual_demand = None

        if (
            "actual_demand_mw"
            in df.columns
            and pd.notna(
                row[
                    "actual_demand_mw"
                ]
            )
        ):
            actual_demand = round(
                float(
                    row[
                        "actual_demand_mw"
                    ]
                ),
                2,
            )

        forecasts.append(
            {
                "forecast_date":
                    str(
                        pd.Timestamp(
                            row[
                                "forecast_date"
                            ]
                        ).date()
                    ),

                "horizon_day":
                    horizon_day,

                "forecast_mode":
                    mode,

                "display_label":
                    display_label,

                "predicted_demand_mw":
                    round(
                        float(
                            row[
                                "predicted_demand_mw"
                            ]
                        ),
                        2,
                    ),

                "actual_demand_mw":
                    actual_demand,

                "actual_available":
                    bool(
                        actual_demand
                        is not None
                    ),

                "forecast_status":
                    str(
                        row[
                            "forecast_status"
                        ]
                    ),

                "validated_horizon":
                    bool(
                        row[
                            "validated_horizon"
                        ]
                    ),

                "model_name":
                    str(
                        row[
                            "model_name"
                        ]
                    ),

                "model_alias":
                    (
                        None
                        if pd.isna(
                            row[
                                "model_alias"
                            ]
                        )
                        else str(
                            row[
                                "model_alias"
                            ]
                        )
                    ),

                "confidence_note":
                    confidence_note,
            }
        )

    return {
        "region":
            region,

        "prediction_type":
            "extended_forecast",

        "latest_real_bpdb_date":
            str(
                latest_real_date.date()
            ),

        "validated_max_horizon_days":
            MAX_BRIDGE_HORIZON,

        "forecast_start":
            forecasts[0][
                "forecast_date"
            ],

        "forecast_end":
            forecasts[-1][
                "forecast_date"
            ],

        "forecast_count":
            len(
                forecasts
            ),

        "actual_values_available":
            any(
                item[
                    "actual_available"
                ]
                for item in forecasts
            ),

        "forecast_policy": {
            "day_1":
                "anchor",

            "day_2_to_day_8":
                "bridge",

            "beyond_day_8":
                "not_served",
        },

        "forecasts":
            forecasts,
    }


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
def startup_event():
    logger.info(
        "Starting forecasting API"
    )

    setup_mlflow()

    load_registered_model(
        NATIONAL_MODEL_NAME
    )

    logger.info(
        "API startup completed"
    )


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():
    return {
        "message":
            (
                "Bangladesh Electricity "
                "Demand Forecasting API"
            ),

        "version":
            "5.0.0",

        "docs":
            "/docs",

        "features": [
            "National forecasting",
            "Regional forecasting",
            "Historical evaluation",
            "Live next-day forecasting",
            "Extended Bridge forecasting",
        ],
    }


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():
    return {
        "status":
            "healthy",

        "national_model":
            (
                "loaded"
                if NATIONAL_MODEL_NAME
                in model_cache
                else
                "not_loaded"
            ),

        "model_registry":
            "MLflow",

        "live_inference":
            True,

        "national_bridge_forecast":
            (
                "available"
                if (
                    NATIONAL_BRIDGE_FORECAST_PATH
                    .exists()
                )
                else
                "unavailable"
            ),

        "regional_bridge_forecast":
            (
                "available"
                if (
                    REGIONAL_BRIDGE_FORECAST_PATH
                    .exists()
                )
                else
                "unavailable"
            ),
    }


# ============================================================
# Regions
# ============================================================

@app.get("/regions")
def get_regions():
    results = []

    for region in REGIONS:
        if region == "National":
            anchor_strategy = "ridge"
            bridge_available = True

        else:
            anchor_strategy = (
                REGIONAL_STRATEGIES[
                    region
                ]
            )

            bridge_available = True

        results.append(
            {
                "region":
                    region,

                "anchor_strategy":
                    anchor_strategy,

                "extended_forecast":
                    bridge_available,
            }
        )

    return {
        "count":
            len(REGIONS),

        "regions":
            results,
    }


# ============================================================
# Latest Live Prediction
# ============================================================

@app.get("/predict/latest")
def predict_latest(
    region: str = Query(
        default="National"
    ),
):
    try:
        validate_region(
            region
        )

        if region == "National":
            df = (
                load_national_inference_data()
            )

        else:
            df = (
                load_regional_inference_data(
                    region
                )
            )

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No live inference "
                    "features available"
                ),
            )

        latest_row = (
            df.iloc[-1]
        )

        return (
            build_live_prediction_response(
                latest_row,
                region,
            )
        )

    except HTTPException:
        raise

    except Exception as error:
        logger.exception(
            "Latest live prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ============================================================
# Extended Forecast
# ============================================================

@app.get("/predict/extended")
def predict_extended(
    region: str = Query(
        default="National"
    ),
):
    try:
        validate_region(
            region
        )

        if region == "National":
            df = (
                load_national_bridge_data()
            )

        else:
            df = (
                load_regional_bridge_data(
                    region
                )
            )

        return (
            build_extended_response(
                df=df,
                region=region,
            )
        )

    except HTTPException:
        raise

    except Exception as error:
        logger.exception(
            f"Extended prediction failed "
            f"for {region}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ============================================================
# Historical Date Prediction
# ============================================================

@app.get("/predict/date")
def predict_date(
    forecast_date: date,

    region: str = Query(
        default="National"
    ),
):
    try:
        validate_region(
            region
        )

        requested_date = (
            pd.Timestamp(
                forecast_date
            )
        )

        if region == "National":
            df = (
                load_national_data()
            )

        else:
            df = (
                load_regional_data(
                    region
                )
            )

        selected = df[
            df[
                "forecast_date"
            ]
            == requested_date
        ]

        if selected.empty:
            raise HTTPException(
                status_code=404,
                detail={
                    "message":
                        (
                            "No historical "
                            "prediction data "
                            "available for "
                            "selected date"
                        ),

                    "requested_date":
                        str(
                            forecast_date
                        ),

                    "available_start":
                        str(
                            df[
                                "forecast_date"
                            ]
                            .min()
                            .date()
                        ),

                    "available_end":
                        str(
                            df[
                                "forecast_date"
                            ]
                            .max()
                            .date()
                        ),
                },
            )

        return (
            build_prediction_response(
                selected.iloc[0],
                region,
            )
        )

    except HTTPException:
        raise

    except Exception as error:
        logger.exception(
            "Historical date prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ============================================================
# History
# ============================================================

@app.get("/history")
def history(
    region: str = Query(
        default="National"
    ),

    limit: int = Query(
        default=90,
        ge=1,
        le=365,
    ),
):
    try:
        validate_region(
            region
        )

        if region == "National":
            df = (
                load_national_data()
            )

        else:
            df = (
                load_regional_data(
                    region
                )
            )

        df = (
            df.tail(limit)
            .copy()
        )

        records = []

        for _, row in df.iterrows():
            result = (
                build_prediction_response(
                    row,
                    region,
                )
            )

            records.append(
                {
                    "region":
                        region,

                    "date":
                        result[
                            "forecast_date"
                        ],

                    "actual_demand_mw":
                        result[
                            "actual_demand_mw"
                        ],

                    "predicted_demand_mw":
                        result[
                            "predicted_demand_mw"
                        ],

                    "absolute_error_mw":
                        result[
                            "absolute_error_mw"
                        ],

                    "percentage_error":
                        result[
                            "percentage_error"
                        ],

                    "evaluation_scope":
                        result[
                            "evaluation_scope"
                        ],
                }
            )

        return {
            "region":
                region,

            "count":
                len(records),

            "records":
                records,
        }

    except HTTPException:
        raise

    except Exception as error:
        logger.exception(
            "History request failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ============================================================
# Model Info
# ============================================================

@app.get("/model/info")
def model_info(
    region: str = Query(
        default="National"
    ),
):
    validate_region(
        region
    )

    if region == "National":
        return {
            "region":
                "National",

            "anchor": {
                "strategy":
                    "ridge",

                "model_name":
                    NATIONAL_MODEL_NAME,

                "alias":
                    MODEL_ALIAS,

                "alpha":
                    0.01,

                "cv_mean_mape":
                    4.8905,

                "holdout_mape":
                    5.68,
            },

            "bridge": {
                "enabled":
                    True,

                "model_name":
                    NATIONAL_BRIDGE_MODEL_NAME,

                "alias":
                    MODEL_ALIAS,

                "alpha":
                    0.01,

                "validated_max_horizon_days":
                    MAX_BRIDGE_HORIZON,

                "backtest_mape":
                    7.01,
            },
        }

    anchor_strategy = (
        REGIONAL_STRATEGIES[
            region
        ]
    )

    return {
        "region":
            region,

        "anchor": {
            "strategy":
                anchor_strategy,

            "model_name":
                (
                    REGIONAL_MODEL_NAMES.get(
                        region
                    )
                ),

            "alias":
                (
                    MODEL_ALIAS
                    if anchor_strategy
                    == "ridge"
                    else None
                ),

            "alpha":
                (
                    0.01
                    if anchor_strategy
                    == "ridge"
                    else None
                ),
        },

        "bridge": {
            "enabled":
                True,

            "model_name":
                (
                    REGIONAL_BRIDGE_MODEL_NAMES[
                        region
                    ]
                ),

            "alias":
                MODEL_ALIAS,

            "alpha":
                0.01,

            "validated_max_horizon_days":
                MAX_BRIDGE_HORIZON,
        },
    }