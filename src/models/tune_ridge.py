import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger


logger = get_logger("tune_ridge")

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

ALPHAS = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]


def calculate_mape(y_true, y_pred):
    return (
        np.mean(
            np.abs(
                (y_true - y_pred) / y_true
            )
        )
        * 100
    )


def main():
    logger.info(
        "Loading feature dataset"
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

    # Keep 2026 completely untouched
    development = df[
        df["Date"] <= "2025-12-31"
    ].copy()

    logger.info(
        f"Development rows: {len(development)}"
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

    for alpha in ALPHAS:
        logger.info(
            f"Testing Ridge alpha={alpha}"
        )

        for fold in folds:
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

            model = Pipeline(
                steps=[
                    (
                        "scaler",
                        StandardScaler()
                    ),
                    (
                        "ridge",
                        Ridge(
                            alpha=alpha
                        )
                    ),
                ]
            )

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_val
            )

            mae = mean_absolute_error(
                y_val,
                predictions
            )

            mape = calculate_mape(
                y_val,
                predictions
            )

            results.append(
                {
                    "alpha": alpha,
                    "fold": fold["name"],
                    "mae": mae,
                    "mape": mape,
                }
            )

            logger.info(
                f"alpha={alpha} | "
                f"{fold['name']} | "
                f"MAPE={mape:.4f}%"
            )

    results_df = pd.DataFrame(
        results
    )

    summary = (
        results_df
        .groupby("alpha")
        .agg(
            mean_mae=(
                "mae",
                "mean"
            ),
            mean_mape=(
                "mape",
                "mean"
            ),
            std_mape=(
                "mape",
                "std"
            ),
        )
        .reset_index()
        .sort_values(
            "mean_mape"
        )
    )

    print(
        "\nRIDGE TUNING RESULTS\n"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\nRIDGE TUNING SUMMARY\n"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    best_alpha = (
        summary.iloc[0]["alpha"]
    )

    best_mape = (
        summary.iloc[0]["mean_mape"]
    )

    logger.info(
        f"Best alpha: {best_alpha}"
    )

    logger.info(
        f"Best mean CV MAPE: "
        f"{best_mape:.4f}%"
    )

    results_df.to_csv(
        "artifacts/ridge_tuning_results.csv",
        index=False
    )

    summary.to_csv(
        "artifacts/ridge_tuning_summary.csv",
        index=False
    )

    logger.info(
        "Ridge tuning completed successfully"
    )


if __name__ == "__main__":
    main()