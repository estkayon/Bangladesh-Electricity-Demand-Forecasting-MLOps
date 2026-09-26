import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger


logger = get_logger("final_train_regional")

DATA_PATH = "data/processed/regional_model_features.csv"
ARTIFACT_DIR = Path("artifacts/regional_models")
STRATEGY_PATH = "artifacts/regional_model_strategy.csv"

EXPERIMENT_NAME = "Bangladesh-Electricity-Demand-Forecasting"

RIDGE_ALPHA = 0.01
TARGET_COLUMN = "next_day_demand_mw"

FEATURE_COLUMNS = [
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

RIDGE_REGIONS = [
    "Dhaka",
    "Chittagong",
    "Comilla",
    "Sylhet",
    "Rangpur",
    "Mymensingh",
    "Khulna",
    
]

BASELINE_REGIONS = [
    "Barisal",
    "Rajshahi",
]


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

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    logger.info(
        f"MLflow experiment: {EXPERIMENT_NAME}"
    )


def sanitize_region(region):
    return (
        region.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def build_model():
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "ridge",
                Ridge(
                    alpha=RIDGE_ALPHA
                )
            ),
        ]
    )


def train_and_register_region(
    df,
    region
):
    logger.info(
        f"Preparing final model for: {region}"
    )

    region_df = df[
        df["region"] == region
    ].copy()

    development = region_df[
        region_df["Date"] <= "2025-12-31"
    ].copy()

    if development.empty:
        raise ValueError(
            f"No development data for {region}"
        )

    X_train = development[
        FEATURE_COLUMNS
    ]

    y_train = development[
        TARGET_COLUMN
    ]

    model = build_model()

    logger.info(
        f"Training final Ridge model for: {region}"
    )

    model.fit(
        X_train,
        y_train
    )

    safe_name = sanitize_region(
        region
    )

    local_path = (
        ARTIFACT_DIR
        / f"{safe_name}_final_ridge.pkl"
    )

    joblib.dump(
        model,
        local_path
    )

    logger.info(
        f"Saved local model: {local_path}"
    )

    registered_model_name = (
        f"bangladesh-electricity-demand-"
        f"{safe_name}-ridge"
    )

    run_name = (
        f"final_regional_{safe_name}_ridge"
    )

    logger.info(
        f"Registering MLflow model: "
        f"{registered_model_name}"
    )

    with mlflow.start_run(
        run_name=run_name
    ):
        mlflow.log_param(
            "region",
            region
        )

        mlflow.log_param(
            "model_type",
            "Ridge"
        )

        mlflow.log_param(
            "alpha",
            RIDGE_ALPHA
        )

        mlflow.log_param(
            "feature_count",
            len(FEATURE_COLUMNS)
        )

        mlflow.log_param(
            "training_data_end",
            "2025-12-31"
        )

        mlflow.log_param(
            "forecast_horizon",
            "next_day"
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train.head(5),
            registered_model_name=(
                registered_model_name
            ),
        )

    logger.info(
        f"Registered model successfully: "
        f"{registered_model_name}"
    )

    return {
        "region": region,
        "strategy": "ridge",
        "registered_model_name":
            registered_model_name,
        "local_model_path":
            str(local_path),
    }


def main():
    setup_mlflow()

    logger.info(
        "Loading regional feature dataset"
    )

    df = pd.read_csv(
        DATA_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    strategy_rows = []

    # --------------------------------------------------
    # Ridge regions
    # --------------------------------------------------

    for region in RIDGE_REGIONS:
        result = train_and_register_region(
            df,
            region
        )

        strategy_rows.append(
            result
        )

    # --------------------------------------------------
    # Baseline regions
    # --------------------------------------------------

    for region in BASELINE_REGIONS:
        logger.info(
            f"{region} uses persistence baseline"
        )

        strategy_rows.append(
            {
                "region": region,
                "strategy": "baseline",
                "registered_model_name": "",
                "local_model_path": "",
            }
        )

    # --------------------------------------------------
    # Save deployment strategy
    # --------------------------------------------------

    strategy_df = pd.DataFrame(
        strategy_rows
    )

    strategy_df = (
        strategy_df
        .sort_values("region")
        .reset_index(drop=True)
    )

    strategy_df.to_csv(
        STRATEGY_PATH,
        index=False
    )

    print(
        "\nREGIONAL DEPLOYMENT STRATEGY\n"
    )

    print(
        strategy_df.to_string(
            index=False
        )
    )

    logger.info(
        f"Regional strategy saved to: "
        f"{STRATEGY_PATH}"
    )

    logger.info(
        "Regional final training and "
        "registration completed successfully"
    )


if __name__ == "__main__":
    main()