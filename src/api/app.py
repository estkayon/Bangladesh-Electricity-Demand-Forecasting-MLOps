import os
from datetime import date

import mlflow
import mlflow.sklearn
import pandas as pd

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.logger import get_logger


logger = get_logger("api")


# ==================================================
# Configuration
# ==================================================

NATIONAL_FEATURE_PATH = (
    "data/processed/model_features.csv"
)

REGIONAL_FEATURE_PATH = (
    "data/processed/regional_model_features.csv"
)

NATIONAL_MODEL_NAME = (
    "bangladesh-electricity-demand-ridge"
)

MODEL_ALIAS = "champion"


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


# Regional deployment strategy selected from time-series CV
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


# ==================================================
# App
# ==================================================

app = FastAPI(
    title=(
        "Bangladesh Electricity Demand "
        "Forecasting API"
    ),
    description=(
        "National and regional next-day "
        "electricity demand forecasting API."
    ),
    version="2.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Model cache
model_cache = {}


# ==================================================
# MLflow
# ==================================================

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
    model_name
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


# ==================================================
# Data Loading
# ==================================================

def load_national_data():
    df = pd.read_csv(
        NATIONAL_FEATURE_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df["forecast_date"] = (
        pd.to_datetime(
            df["forecast_date"]
        )
    )

    return (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )


def load_regional_data(
    region=None
):
    df = pd.read_csv(
        REGIONAL_FEATURE_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df["forecast_date"] = (
        pd.to_datetime(
            df["forecast_date"]
        )
    )

    if region:
        df = df[
            df["region"] == region
        ].copy()

    return (
        df
        .sort_values(
            ["region", "Date"]
        )
        .reset_index(drop=True)
    )


# ==================================================
# Validation
# ==================================================

def validate_region(
    region
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
    forecast_date
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


# ==================================================
# Prediction Helpers
# ==================================================

def predict_national_row(
    row
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

    prediction = float(
        model.predict(X)[0]
    )

    return prediction


def predict_regional_row(
    row,
    region
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

    prediction = float(
        model.predict(X)[0]
    )

    return prediction


def build_prediction_response(
    row,
    region
):
    forecast_date = pd.Timestamp(
        row["forecast_date"]
    )

    observation_date = pd.Timestamp(
        row["Date"]
    )

    if region == "National":

        current_demand = float(
            row[
                "total_demand"
            ]
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
                region
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
        "region": region,

        "observation_date": str(
            observation_date.date()
        ),

        "forecast_date": str(
            forecast_date.date()
        ),

        "current_demand_mw": round(
            current_demand,
            2
        ),

        "predicted_demand_mw": round(
            prediction,
            2
        ),

        "actual_demand_mw": round(
            actual_demand,
            2
        ),

        "change_from_previous_day_mw":
            round(
                change,
                2
            ),

        "change_percent":
            round(
                change_percent,
                2
            ),

        "forecast_error_mw":
            round(
                forecast_error,
                2
            ),

        "absolute_error_mw":
            round(
                absolute_error,
                2
            ),

        "percentage_error":
            round(
                percentage_error,
                2
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

            "source":
                (
                    "MLflow Model Registry"
                    if strategy == "ridge"
                    else
                    "Persistence Baseline"
                ),
        },

        "weather": {
            "temperature_max_c":
                float(
                    row[
                        "temperature_max_c"
                    ]
                ),

            "temperature_min_c":
                float(
                    row[
                        "temperature_min_c"
                    ]
                ),

            "temperature_mean_c":
                float(
                    row[
                        "temperature_mean_c"
                    ]
                ),

            "precipitation_mm":
                float(
                    row[
                        "precipitation_mm"
                    ]
                ),

            "rain_mm":
                float(
                    row[
                        "rain_mm"
                    ]
                ),
        },
    }


# ==================================================
# Startup
# ==================================================

@app.on_event("startup")
def startup_event():
    logger.info(
        "Starting forecasting API"
    )

    setup_mlflow()

    # Load national champion at startup
    load_registered_model(
        NATIONAL_MODEL_NAME
    )

    logger.info(
        "API startup completed"
    )


# ==================================================
# Basic Endpoints
# ==================================================

@app.get("/")
def root():
    return {
        "message": (
            "Bangladesh Electricity "
            "Demand Forecasting API"
        ),
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
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
    }


# ==================================================
# Regions
# ==================================================

@app.get("/regions")
def get_regions():
    regional_strategies = []

    for region in REGIONS:

        if region == "National":
            strategy = "ridge"
        else:
            strategy = (
                REGIONAL_STRATEGIES[
                    region
                ]
            )

        regional_strategies.append(
            {
                "region": region,
                "strategy": strategy,
            }
        )

    return {
        "count": len(REGIONS),
        "regions":
            regional_strategies,
    }


# ==================================================
# Latest Prediction
# ==================================================

@app.get("/predict/latest")
def predict_latest(
    region: str = Query(
        default="National"
    )
):
    try:
        validate_region(
            region
        )

        if region == "National":
            df = load_national_data()
        else:
            df = load_regional_data(
                region
            )

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No feature data "
                    "available"
                ),
            )

        latest_row = (
            df.iloc[-1]
        )

        return (
            build_prediction_response(
                latest_row,
                region
            )
        )

    except HTTPException:
        raise

    except Exception as error:
        logger.exception(
            "Latest prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ==================================================
# Date-Specific Prediction
# ==================================================

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
            df = load_national_data()

        else:
            df = load_regional_data(
                region
            )

        selected = df[
            df["forecast_date"]
            == requested_date
        ]

        if selected.empty:
            available_start = str(
                df[
                    "forecast_date"
                ]
                .min()
                .date()
            )

            available_end = str(
                df[
                    "forecast_date"
                ]
                .max()
                .date()
            )

            raise HTTPException(
                status_code=404,
                detail={
                    "message":
                        "No prediction data "
                        "available for the "
                        "selected date",

                    "requested_date":
                        str(
                            forecast_date
                        ),

                    "available_start":
                        available_start,

                    "available_end":
                        available_end,
                },
            )

        row = selected.iloc[0]

        return (
            build_prediction_response(
                row,
                region
            )
        )

    except HTTPException:
        raise

    except Exception as error:
        logger.exception(
            "Date prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ==================================================
# Historical Actual vs Predicted
# ==================================================

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
            df = load_national_data()
        else:
            df = load_regional_data(
                region
            )

        history_df = (
            df.tail(limit)
            .copy()
        )

        records = []

        for _, row in (
            history_df.iterrows()
        ):

            response = (
                build_prediction_response(
                    row,
                    region
                )
            )

            records.append(
                {
                    "region":
                        region,

                    "date":
                        response[
                            "forecast_date"
                        ],

                    "actual_demand_mw":
                        response[
                            "actual_demand_mw"
                        ],

                    "predicted_demand_mw":
                        response[
                            "predicted_demand_mw"
                        ],

                    "absolute_error_mw":
                        response[
                            "absolute_error_mw"
                        ],

                    "percentage_error":
                        response[
                            "percentage_error"
                        ],

                    "evaluation_scope":
                        response[
                            "evaluation_scope"
                        ],
                }
            )

        return {
            "region": region,
            "count": len(records),
            "records": records,
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


# ==================================================
# Model Information
# ==================================================

@app.get("/model/info")
def model_info(
    region: str = Query(
        default="National"
    )
):
    validate_region(
        region
    )

    if region == "National":

        return {
            "region": "National",
            "strategy": "ridge",
            "model_name":
                NATIONAL_MODEL_NAME,
            "alias":
                MODEL_ALIAS,
            "alpha": 0.01,
            "cv_mean_mape": 4.8905,
            "holdout_mape": 5.68,
            "feature_count":
                len(
                    NATIONAL_FEATURE_COLUMNS
                ),
        }

    strategy = (
        REGIONAL_STRATEGIES[
            region
        ]
    )

    return {
        "region": region,
        "strategy": strategy,
        "model_name":
            REGIONAL_MODEL_NAMES.get(
                region
            ),
        "alias":
            (
                MODEL_ALIAS
                if strategy == "ridge"
                else None
            ),
        "alpha":
            (
                0.01
                if strategy == "ridge"
                else None
            ),
        "feature_count":
            len(
                REGIONAL_FEATURE_COLUMNS
            ),
    }