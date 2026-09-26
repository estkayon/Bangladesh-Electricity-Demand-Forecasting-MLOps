import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.logger import get_logger


logger = get_logger("time_series_validation")

DATA_PATH = "data/processed/model_features.csv"
TARGET_COLUMN = "next_day_total_demand"

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


def build_models():
    ridge = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "ridge",
                Ridge(alpha=1.0)
            ),
        ]
    )

    random_forest = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    xgboost = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )

    return {
        "ridge": ridge,
        "random_forest": random_forest,
        "xgboost": xgboost,
    }


def main():
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

    # Keep 2026 completely untouched here
    development = df[
        df["Date"] <= "2025-12-31"
    ].copy()

    logger.info(
        f"Development rows: {len(development)}"
    )

    logger.info(
        f"Development date range: "
        f"{development['Date'].min().date()} "
        f"to "
        f"{development['Date'].max().date()}"
    )

    # Expanding-window folds
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

    for fold in folds:
        logger.info(
            f"Running {fold['name']}"
        )

        train = development[
            development["Date"]
            <= fold["train_end"]
        ].copy()

        validation = development[
            (
                development["Date"]
                >= fold["val_start"]
            )
            &
            (
                development["Date"]
                <= fold["val_end"]
            )
        ].copy()

        logger.info(
            f"{fold['name']} train rows: "
            f"{len(train)}"
        )

        logger.info(
            f"{fold['name']} validation rows: "
            f"{len(validation)}"
        )

        if train.empty or validation.empty:
            logger.warning(
                f"Skipping {fold['name']} "
                f"because train or validation is empty"
            )
            continue

        y_train = train[
            TARGET_COLUMN
        ]

        y_val = validation[
            TARGET_COLUMN
        ]

        X_train = train[
            FEATURE_COLUMNS
        ]

        X_val = validation[
            FEATURE_COLUMNS
        ]

        # ------------------------------------------
        # Baseline
        # ------------------------------------------

        baseline_pred = validation[
            "total_demand"
        ]

        mae, rmse, mape = calculate_metrics(
            y_val,
            baseline_pred
        )

        results.append(
            {
                "fold": fold["name"],
                "model": "baseline",
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
            }
        )

        logger.info(
            f"{fold['name']} baseline MAPE: "
            f"{mape:.2f}%"
        )

        # ------------------------------------------
        # ML Models
        # ------------------------------------------

        models = build_models()

        for model_name, model in models.items():
            logger.info(
                f"Training {model_name} "
                f"on {fold['name']}"
            )

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_val
            )

            mae, rmse, mape = calculate_metrics(
                y_val,
                predictions
            )

            results.append(
                {
                    "fold": fold["name"],
                    "model": model_name,
                    "mae": mae,
                    "rmse": rmse,
                    "mape": mape,
                }
            )

            logger.info(
                f"{fold['name']} "
                f"{model_name} MAPE: "
                f"{mape:.2f}%"
            )

    results_df = pd.DataFrame(
        results
    )

    print("\nFOLD RESULTS\n")

    print(
        results_df.to_string(
            index=False
        )
    )

    summary = (
        results_df
        .groupby("model")
        .agg(
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mape=("mape", "mean"),
            std_mape=("mape", "std"),
        )
        .reset_index()
        .sort_values(
            "mean_mape"
        )
    )

    print("\nCROSS-VALIDATION SUMMARY\n")

    print(
        summary.to_string(
            index=False
        )
    )

    results_df.to_csv(
        "artifacts/time_series_cv_results.csv",
        index=False
    )

    summary.to_csv(
        "artifacts/time_series_cv_summary.csv",
        index=False
    )

    logger.info(
        "Time-series validation results saved"
    )

    logger.info(
        "Time-series validation completed successfully"
    )


if __name__ == "__main__":
    main()