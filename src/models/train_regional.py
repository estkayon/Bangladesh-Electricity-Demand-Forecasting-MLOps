import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger


logger = get_logger("train_regional")

DATA_PATH = "data/processed/regional_model_features.csv"
ARTIFACT_DIR = Path("artifacts/regional_models")
METRICS_PATH = "artifacts/regional_model_metrics.csv"

TARGET_COLUMN = "next_day_demand_mw"
RIDGE_ALPHA = 0.01

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


def sanitize_region_name(region):
    return (
        region.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def train_region(region_df, region_name):
    logger.info(
        f"Training regional model for: {region_name}"
    )

    development = region_df[
        region_df["Date"] <= "2025-12-31"
    ].copy()

    holdout = region_df[
        region_df["Date"] >= "2026-01-01"
    ].copy()

    logger.info(
        f"{region_name} development rows: "
        f"{len(development)}"
    )

    logger.info(
        f"{region_name} holdout rows: "
        f"{len(holdout)}"
    )

    if development.empty:
        raise ValueError(
            f"No development data for {region_name}"
        )

    if holdout.empty:
        raise ValueError(
            f"No holdout data for {region_name}"
        )

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

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_holdout
    )

    mae, rmse, mape = calculate_metrics(
        y_holdout,
        predictions
    )

    safe_name = sanitize_region_name(
        region_name
    )

    model_path = (
        ARTIFACT_DIR
        / f"{safe_name}_ridge_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    logger.info(
        f"{region_name} model saved to: "
        f"{model_path}"
    )

    logger.info(
        f"{region_name} | "
        f"MAE: {mae:.2f} MW | "
        f"RMSE: {rmse:.2f} MW | "
        f"MAPE: {mape:.2f}%"
    )

    return {
        "region": region_name,
        "development_rows": len(development),
        "holdout_rows": len(holdout),
        "mae_mw": mae,
        "rmse_mw": rmse,
        "mape_percent": mape,
        "alpha": RIDGE_ALPHA,
        "model_path": str(model_path),
    }


def main():
    logger.info(
        "Loading regional model feature dataset"
    )

    df = pd.read_csv(
        DATA_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = (
        df
        .sort_values(
            ["region", "Date"]
        )
        .reset_index(drop=True)
    )

    regions = sorted(
        df["region"].unique()
    )

    logger.info(
        f"Regions found: {len(regions)}"
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    for region in regions:
        region_df = df[
            df["region"] == region
        ].copy()

        result = train_region(
            region_df,
            region
        )

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    ).sort_values(
        "mape_percent"
    )

    print(
        "\nREGIONAL MODEL RESULTS\n"
    )

    print(
        results_df[
            [
                "region",
                "mae_mw",
                "rmse_mw",
                "mape_percent",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nAVERAGE REGIONAL PERFORMANCE\n"
    )

    print(
        f"Mean MAE: "
        f"{results_df['mae_mw'].mean():.2f} MW"
    )

    print(
        f"Mean RMSE: "
        f"{results_df['rmse_mw'].mean():.2f} MW"
    )

    print(
        f"Mean MAPE: "
        f"{results_df['mape_percent'].mean():.2f}%"
    )

    results_df.to_csv(
        METRICS_PATH,
        index=False
    )

    logger.info(
        f"Regional metrics saved to: "
        f"{METRICS_PATH}"
    )

    logger.info(
        "Regional model training completed successfully"
    )


if __name__ == "__main__":
    main()