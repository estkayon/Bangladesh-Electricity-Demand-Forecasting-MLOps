import os

import mlflow
import pandas as pd

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.logger import get_logger


logger = get_logger("api")

FEATURE_DATA_PATH = "data/processed/model_features.csv"

MODEL_NAME = "bangladesh-electricity-demand-ridge"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

FEATURE_COLUMNS = [
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

MODEL_INFO = {
    "model_name": MODEL_NAME,
    "model_type": "Ridge Regression",
    "model_alias": MODEL_ALIAS,
    "model_uri": MODEL_URI,
    "alpha": 0.01,
    "scaler": "StandardScaler",
    "forecast_horizon": "Next-day",
    "target": "National electricity demand",
    "unit": "MW",
    "feature_count": len(FEATURE_COLUMNS),
    "cv_mean_mape": 4.8905,
    "holdout_mape": 5.68,
    "holdout_mae_mw": 743.59,
    "holdout_rmse_mw": 1027.79,
}


app = FastAPI(
    title="Bangladesh Electricity Demand Forecasting API",
    description=(
        "Production-oriented API for next-day national "
        "electricity demand forecasting in Bangladesh."
    ),
    version="1.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None


def setup_mlflow():
    logger.info(
        "Loading MLflow configuration"
    )

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


def load_registered_model():
    global model

    setup_mlflow()

    logger.info(
        f"Loading registered model: {MODEL_URI}"
    )

    model = mlflow.sklearn.load_model(
        MODEL_URI
    )

    logger.info(
        "Registered champion model loaded successfully"
    )


def load_feature_data():
    logger.info(
        "Loading feature dataset"
    )

    df = pd.read_csv(
        FEATURE_DATA_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"]
    )

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    return df


def prepare_latest_features():
    df = load_feature_data()

    if df.empty:
        raise ValueError(
            "Feature dataset is empty"
        )

    latest_row = (
        df.iloc[-1]
        .copy()
    )

    missing_columns = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in latest_row.index
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: "
            f"{missing_columns}"
        )

    feature_values = latest_row[
        FEATURE_COLUMNS
    ]

    if feature_values.isna().any():
        missing_features = (
            feature_values[
                feature_values.isna()
            ]
            .index
            .tolist()
        )

        raise ValueError(
            f"Missing feature values: "
            f"{missing_features}"
        )

    X_latest = pd.DataFrame(
        [
            feature_values.to_dict()
        ]
    )

    return latest_row, X_latest


@app.on_event("startup")
def startup_event():
    logger.info(
        "Starting electricity demand forecasting API"
    )

    load_registered_model()

    logger.info(
        "API startup completed"
    )


@app.get("/")
def root():
    return {
        "message": (
            "Bangladesh Electricity Demand "
            "Forecasting API"
        ),
        "version": "1.1.0",
        "model_source": "MLflow Model Registry",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_status": (
            "loaded"
            if model is not None
            else "not_loaded"
        ),
        "model_source": "mlflow_registry",
        "model_alias": MODEL_ALIAS,
    }


@app.get("/model/info")
def model_info():
    return MODEL_INFO


@app.get("/predict/latest")
def predict_latest():
    try:
        if model is None:
            load_registered_model()

        latest_row, X_latest = (
            prepare_latest_features()
        )

        logger.info(
            "Generating latest next-day prediction"
        )

        prediction = float(
            model.predict(
                X_latest
            )[0]
        )

        current_demand = float(
            latest_row["total_demand"]
        )

        difference = (
            prediction
            - current_demand
        )

        percentage_change = (
            (difference / current_demand) * 100
            if current_demand != 0
            else 0.0
        )

        logger.info(
            f"Prediction completed: "
            f"{prediction:.2f} MW"
        )

        return {
            "observation_date": str(
                latest_row["Date"].date()
            ),
            "forecast_date": str(
                latest_row[
                    "forecast_date"
                ].date()
            ),
            "current_demand_mw": round(
                current_demand,
                2
            ),
            "predicted_demand_mw": round(
                prediction,
                2
            ),
            "change_mw": round(
                difference,
                2
            ),
            "change_percent": round(
                percentage_change,
                2
            ),
            "model": {
                "name": MODEL_NAME,
                "alias": MODEL_ALIAS,
                "source": "MLflow Model Registry",
            },
            "weather": {
                "temperature_max_c": float(
                    latest_row[
                        "temperature_max_c"
                    ]
                ),
                "temperature_min_c": float(
                    latest_row[
                        "temperature_min_c"
                    ]
                ),
                "temperature_mean_c": float(
                    latest_row[
                        "temperature_mean_c"
                    ]
                ),
                "precipitation_mm": float(
                    latest_row[
                        "precipitation_mm"
                    ]
                ),
                "rain_mm": float(
                    latest_row[
                        "rain_mm"
                    ]
                ),
            },
        }

    except Exception as error:
        logger.exception(
            "Prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.get("/history")
def history(limit: int = 30):
    try:
        if limit < 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "limit must be greater than 0"
                ),
            )

        limit = min(
            limit,
            365
        )

        df = load_feature_data()

        history_df = (
            df.tail(limit)
            .copy()
        )

        records = []

        for _, row in history_df.iterrows():
            records.append(
                {
                    "date": str(
                        row["Date"].date()
                    ),
                    "forecast_date": str(
                        row[
                            "forecast_date"
                        ].date()
                    ),
                    "total_demand_mw": float(
                        row[
                            "total_demand"
                        ]
                    ),
                    "next_day_actual_demand_mw": float(
                        row[
                            "next_day_total_demand"
                        ]
                    ),
                    "temperature_mean_c": float(
                        row[
                            "temperature_mean_c"
                        ]
                    ),
                    "is_holiday": int(
                        row[
                            "is_holiday"
                        ]
                    ),
                    "is_weekend": int(
                        row[
                            "is_weekend"
                        ]
                    ),
                }
            )

        return {
            "count": len(records),
            "records": records,
        }

    except HTTPException:
        raise

    except Exception as error:
        logger.exception(
            "Failed to load history"
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )