import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger


logger = get_logger("final_train")

DATA_PATH = "data/processed/model_features.csv"
MODEL_OUTPUT_PATH = "artifacts/final_ridge_model.pkl"
FEATURES_OUTPUT_PATH = "artifacts/final_ridge_features.txt"

EXPERIMENT_NAME = "Bangladesh-Electricity-Demand-Forecasting"
RUN_NAME = "final_ridge_model"

TARGET_COLUMN = "next_day_total_demand"
RIDGE_ALPHA = 0.01

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


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mape = (
        np.mean(
            np.abs(
                (y_true - y_pred)
                / y_true
            )
        )
        * 100
    )

    return mae, rmse, mape


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


def load_data():
    logger.info(
        "Loading model feature dataset"
    )

    df = pd.read_csv(
        DATA_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    development = df[
        df["Date"] <= "2025-12-31"
    ].copy()

    holdout = df[
        df["Date"] >= "2026-01-01"
    ].copy()

    logger.info(
        f"Development rows: {len(development)}"
    )

    logger.info(
        f"Holdout rows: {len(holdout)}"
    )

    logger.info(
        f"Development range: "
        f"{development['Date'].min().date()} "
        f"to "
        f"{development['Date'].max().date()}"
    )

    logger.info(
        f"Holdout range: "
        f"{holdout['Date'].min().date()} "
        f"to "
        f"{holdout['Date'].max().date()}"
    )

    return development, holdout


def build_model():
    logger.info(
        f"Building Ridge model with alpha={RIDGE_ALPHA}"
    )

    model = Pipeline(
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

    return model


def save_feature_list():
    logger.info(
        "Saving model feature list"
    )

    with open(
        FEATURES_OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        for feature in FEATURE_COLUMNS:
            file.write(
                f"{feature}\n"
            )


def main():
    setup_mlflow()

    development, holdout = load_data()

    X_train = development[
        FEATURE_COLUMNS
    ]

    y_train = development[
        TARGET_COLUMN
    ]

    X_holdout = holdout[
        FEATURE_COLUMNS
    ]

    y_holdout = holdout[
        TARGET_COLUMN
    ]

    model = build_model()

    logger.info(
        "Training final Ridge model "
        "using development data"
    )

    model.fit(
        X_train,
        y_train
    )

    logger.info(
        "Final Ridge training completed"
    )

    logger.info(
        "Evaluating final model on 2026 holdout"
    )

    predictions = model.predict(
        X_holdout
    )

    mae, rmse, mape = calculate_metrics(
        y_holdout,
        predictions
    )

    logger.info(
        "Final Holdout Results"
    )

    logger.info(
        f"MAE: {mae:.2f} MW"
    )

    logger.info(
        f"RMSE: {rmse:.2f} MW"
    )

    logger.info(
        f"MAPE: {mape:.2f}%"
    )

    os.makedirs(
        "artifacts",
        exist_ok=True
    )

    logger.info(
        "Saving final model locally"
    )

    joblib.dump(
        model,
        MODEL_OUTPUT_PATH
    )

    logger.info(
        f"Model saved to: "
        f"{MODEL_OUTPUT_PATH}"
    )

    save_feature_list()

    logger.info(
        f"Feature list saved to: "
        f"{FEATURES_OUTPUT_PATH}"
    )

    logger.info(
        "Logging final model to MLflow"
    )

    with mlflow.start_run(
        run_name=RUN_NAME
    ):
        mlflow.log_param(
            "model_type",
            "Ridge"
        )

        mlflow.log_param(
            "alpha",
            RIDGE_ALPHA
        )

        mlflow.log_param(
            "scaler",
            "StandardScaler"
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
            "holdout_start",
            "2026-01-01"
        )

        mlflow.log_param(
            "forecast_horizon",
            "next_day"
        )

        mlflow.log_metrics(
            {
                "holdout_mae": mae,
                "holdout_rmse": rmse,
                "holdout_mape": mape,
                "cv_mean_mape": 4.8905,
            }
        )

        mlflow.log_artifact(
            FEATURES_OUTPUT_PATH
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )

    logger.info(
        "Final model logged to MLflow successfully"
    )

    logger.info(
        "Final training pipeline completed successfully"
    )


if __name__ == "__main__":
    main()