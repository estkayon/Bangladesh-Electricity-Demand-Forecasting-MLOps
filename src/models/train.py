import os

import mlflow
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src.logger import get_logger


logger = get_logger("train")

DATA_PATH = "data/processed/model_features.csv"
EXPERIMENT_NAME = "Bangladesh-Electricity-Demand-Forecasting"


FEATURE_COLUMNS = [
    "total_demand",
    "total_load_shed",
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
    "lag_1_day",
    "lag_7_day",
    "lag_14_day",
    "rolling_7_day_mean",
    "rolling_14_day_mean",
    "rolling_30_day_mean",
    "load_shed_lag_1_day",
]

TARGET_COLUMN = "next_day_total_demand"


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    mape = (
        np.mean(
            np.abs(
                (y_true - y_pred) / y_true
            )
        )
        * 100
    )

    return mae, rmse, mape


def setup_mlflow():
    logger.info("Loading MLflow configuration")

    load_dotenv()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI not found")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info(
        f"MLflow experiment: {EXPERIMENT_NAME}"
    )


def load_and_split_data():
    logger.info("Loading feature dataset")

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    logger.info(
        "Creating chronological train/validation/test splits"
    )

    train = df[
        df["Date"] <= "2024-12-31"
    ].copy()

    validation = df[
        (df["Date"] >= "2025-01-01")
        & (df["Date"] <= "2025-12-31")
    ].copy()

    test = df[
        df["Date"] >= "2026-01-01"
    ].copy()

    logger.info(f"Train rows: {len(train)}")
    logger.info(f"Validation rows: {len(validation)}")
    logger.info(f"Test rows: {len(test)}")

    return train, validation, test


def run_baseline(validation, test):
    logger.info("Running naive persistence baseline")

    y_val = validation[TARGET_COLUMN]
    val_pred = validation["total_demand"]

    y_test = test[TARGET_COLUMN]
    test_pred = test["total_demand"]

    val_mae, val_rmse, val_mape = calculate_metrics(
        y_val,
        val_pred
    )

    test_mae, test_rmse, test_mape = calculate_metrics(
        y_test,
        test_pred
    )

    with mlflow.start_run(
        run_name="naive_persistence_baseline"
    ):
        mlflow.log_param(
            "model_type",
            "naive_persistence"
        )

        mlflow.log_param(
            "forecast_horizon",
            "next_day"
        )

        mlflow.log_metrics(
            {
                "validation_mae": val_mae,
                "validation_rmse": val_rmse,
                "validation_mape": val_mape,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_mape": test_mape,
            }
        )

    logger.info("Baseline Validation Results")
    logger.info(f"MAE: {val_mae:.2f} MW")
    logger.info(f"RMSE: {val_rmse:.2f} MW")
    logger.info(f"MAPE: {val_mape:.2f}%")

    logger.info("Baseline Test Results")
    logger.info(f"MAE: {test_mae:.2f} MW")
    logger.info(f"RMSE: {test_rmse:.2f} MW")
    logger.info(f"MAPE: {test_mape:.2f}%")

    logger.info(
        "Baseline experiment logged to MLflow"
    )


def run_random_forest(train, validation, test):
    logger.info("Preparing Random Forest experiment")

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    X_val = validation[FEATURE_COLUMNS]
    y_val = validation[TARGET_COLUMN]

    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    rf_params = {
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    }

    logger.info("Training Random Forest model")

    model = RandomForestRegressor(
        **rf_params
    )

    model.fit(
        X_train,
        y_train
    )

    logger.info(
        "Random Forest training completed"
    )

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_mae, val_rmse, val_mape = calculate_metrics(
        y_val,
        val_pred
    )

    test_mae, test_rmse, test_mape = calculate_metrics(
        y_test,
        test_pred
    )

    with mlflow.start_run(
        run_name="random_forest_v1"
    ):
        mlflow.log_param(
            "model_type",
            "RandomForestRegressor"
        )

        mlflow.log_params(rf_params)

        mlflow.log_param(
            "feature_count",
            len(FEATURE_COLUMNS)
        )

        mlflow.log_metrics(
            {
                "validation_mae": val_mae,
                "validation_rmse": val_rmse,
                "validation_mape": val_mape,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_mape": test_mape,
            }
        )

    logger.info(
        "Random Forest Validation Results"
    )
    logger.info(
        f"MAE: {val_mae:.2f} MW"
    )
    logger.info(
        f"RMSE: {val_rmse:.2f} MW"
    )
    logger.info(
        f"MAPE: {val_mape:.2f}%"
    )

    logger.info(
        "Random Forest Test Results"
    )
    logger.info(
        f"MAE: {test_mae:.2f} MW"
    )
    logger.info(
        f"RMSE: {test_rmse:.2f} MW"
    )
    logger.info(
        f"MAPE: {test_mape:.2f}%"
    )

    logger.info(
        "Random Forest experiment logged to MLflow"
    )


def run_xgboost(train, validation, test):
    logger.info("Preparing XGBoost experiment")

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    X_val = validation[FEATURE_COLUMNS]
    y_val = validation[TARGET_COLUMN]

    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    xgb_params = {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "objective": "reg:squarederror",
    }

    logger.info("Training XGBoost model")

    model = XGBRegressor(
        **xgb_params
    )

    model.fit(
        X_train,
        y_train
    )

    logger.info(
        "XGBoost training completed"
    )

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_mae, val_rmse, val_mape = calculate_metrics(
        y_val,
        val_pred
    )

    test_mae, test_rmse, test_mape = calculate_metrics(
        y_test,
        test_pred
    )

    with mlflow.start_run(
        run_name="xgboost_v1"
    ):
        mlflow.log_param(
            "model_type",
            "XGBRegressor"
        )

        mlflow.log_params(
            xgb_params
        )

        mlflow.log_param(
            "feature_count",
            len(FEATURE_COLUMNS)
        )

        mlflow.log_metrics(
            {
                "validation_mae": val_mae,
                "validation_rmse": val_rmse,
                "validation_mape": val_mape,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_mape": test_mape,
            }
        )

    logger.info(
        "XGBoost Validation Results"
    )
    logger.info(
        f"MAE: {val_mae:.2f} MW"
    )
    logger.info(
        f"RMSE: {val_rmse:.2f} MW"
    )
    logger.info(
        f"MAPE: {val_mape:.2f}%"
    )

    logger.info(
        "XGBoost Test Results"
    )
    logger.info(
        f"MAE: {test_mae:.2f} MW"
    )
    logger.info(
        f"RMSE: {test_rmse:.2f} MW"
    )
    logger.info(
        f"MAPE: {test_mape:.2f}%"
    )

    logger.info(
        "XGBoost experiment logged to MLflow"
    )


def main():
    setup_mlflow()

    train, validation, test = (
        load_and_split_data()
    )

    run_baseline(
        validation,
        test
    )

    run_random_forest(
        train,
        validation,
        test
    )

    run_xgboost(
        train,
        validation,
        test
    )


if __name__ == "__main__":
    main()