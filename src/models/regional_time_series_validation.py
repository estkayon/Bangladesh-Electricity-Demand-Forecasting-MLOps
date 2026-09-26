import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger


logger = get_logger("regional_time_series_validation")

DATA_PATH = "data/processed/regional_model_features.csv"
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

RIDGE_ALPHA = 0.01


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
                (y_true - y_pred) / y_true
            )
        )
        * 100
    )

    return mae, rmse, mape


def build_ridge():
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


def main():
    logger.info(
        "Loading regional feature dataset"
    )

    df = pd.read_csv(
        DATA_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = (
        df.sort_values(
            ["region", "Date"]
        )
        .reset_index(drop=True)
    )

    development = df[
        df["Date"] <= "2025-12-31"
    ].copy()

    regions = sorted(
        development[
            "region"
        ].unique()
    )

    folds = [
        {
            "name": "fold_1",
            "train_end": "2022-12-31",
            "val_start": "2023-01-01",
            "val_end": "2023-12-31",
        },
        {
            "name": "fold_2",
            "train_end": "2023-12-31",
            "val_start": "2024-01-01",
            "val_end": "2024-12-31",
        },
        {
            "name": "fold_3",
            "train_end": "2024-12-31",
            "val_start": "2025-01-01",
            "val_end": "2025-12-31",
        },
    ]

    results = []

    for region in regions:
        logger.info(
            f"Validating region: {region}"
        )

        region_df = development[
            development["region"] == region
        ].copy()

        for fold in folds:
            train = region_df[
                region_df["Date"]
                <= fold["train_end"]
            ].copy()

            validation = region_df[
                (
                    region_df["Date"]
                    >= fold["val_start"]
                )
                &
                (
                    region_df["Date"]
                    <= fold["val_end"]
                )
            ].copy()

            if train.empty or validation.empty:
                logger.warning(
                    f"Skipping {region} {fold['name']} "
                    f"because train/validation is empty"
                )
                continue

            X_train = train[
                FEATURE_COLUMNS
            ]

            y_train = train[
                TARGET_COLUMN
            ]

            X_val = validation[
                FEATURE_COLUMNS
            ]

            y_val = validation[
                TARGET_COLUMN
            ]

            # ----------------------------------
            # Naive persistence baseline
            # ----------------------------------

            baseline_pred = validation[
                "regional_demand_mw"
            ]

            baseline_mae, baseline_rmse, baseline_mape = (
                calculate_metrics(
                    y_val,
                    baseline_pred
                )
            )

            results.append(
                {
                    "region": region,
                    "fold": fold["name"],
                    "model": "baseline",
                    "mae": baseline_mae,
                    "rmse": baseline_rmse,
                    "mape": baseline_mape,
                }
            )

            # ----------------------------------
            # Ridge
            # ----------------------------------

            ridge = build_ridge()

            ridge.fit(
                X_train,
                y_train
            )

            ridge_pred = ridge.predict(
                X_val
            )

            ridge_mae, ridge_rmse, ridge_mape = (
                calculate_metrics(
                    y_val,
                    ridge_pred
                )
            )

            results.append(
                {
                    "region": region,
                    "fold": fold["name"],
                    "model": "ridge",
                    "mae": ridge_mae,
                    "rmse": ridge_rmse,
                    "mape": ridge_mape,
                }
            )

            logger.info(
                f"{region} | "
                f"{fold['name']} | "
                f"Baseline MAPE: "
                f"{baseline_mape:.2f}% | "
                f"Ridge MAPE: "
                f"{ridge_mape:.2f}%"
            )

    results_df = pd.DataFrame(
        results
    )

    # ------------------------------------------
    # Per-region summary
    # ------------------------------------------

    regional_summary = (
        results_df
        .groupby(
            ["region", "model"]
        )
        .agg(
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mape=("mape", "mean"),
            std_mape=("mape", "std"),
        )
        .reset_index()
    )

    print(
        "\nREGIONAL CROSS-VALIDATION SUMMARY\n"
    )

    print(
        regional_summary.to_string(
            index=False
        )
    )

    # ------------------------------------------
    # Baseline vs Ridge comparison
    # ------------------------------------------

    comparison = (
        regional_summary
        .pivot(
            index="region",
            columns="model",
            values="mean_mape"
        )
        .reset_index()
    )

    comparison[
        "ridge_improvement_percent"
    ] = (
        comparison["baseline"]
        - comparison["ridge"]
    )

    comparison[
        "recommended_model"
    ] = np.where(
        comparison["ridge"]
        < comparison["baseline"],
        "ridge",
        "baseline"
    )

    comparison = (
        comparison
        .sort_values(
            "ridge_improvement_percent",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(
        "\nREGIONAL MODEL COMPARISON\n"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    # ------------------------------------------
    # Overall summary
    # ------------------------------------------

    overall_summary = (
        results_df
        .groupby("model")
        .agg(
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mape=("mape", "mean"),
            std_mape=("mape", "std"),
        )
        .reset_index()
        .sort_values("mean_mape")
    )

    print(
        "\nOVERALL REGIONAL CV SUMMARY\n"
    )

    print(
        overall_summary.to_string(
            index=False
        )
    )

    results_df.to_csv(
        "artifacts/regional_cv_results.csv",
        index=False
    )

    regional_summary.to_csv(
        "artifacts/regional_cv_summary.csv",
        index=False
    )

    comparison.to_csv(
        "artifacts/regional_model_comparison.csv",
        index=False
    )

    logger.info(
        "Regional time-series validation "
        "completed successfully"
    )


if __name__ == "__main__":
    main()