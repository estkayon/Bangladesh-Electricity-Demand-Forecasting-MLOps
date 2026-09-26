import os
from pathlib import Path

import holidays
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger


logger = get_logger("train_bridge_model")


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = (
    "data/processed/area_wise_demand_processed.csv"
)

ARTIFACT_DIR = Path(
    "artifacts/bridge_model"
)

MODEL_PATH = (
    ARTIFACT_DIR
    / "bridge_ridge_model.pkl"
)

FEATURE_PATH = (
    ARTIFACT_DIR
    / "bridge_features.txt"
)

TRAINING_INFO_PATH = (
    ARTIFACT_DIR
    / "bridge_training_info.csv"
)


EXPERIMENT_NAME = (
    "Bangladesh-Electricity-Demand-Forecasting"
)

REGISTERED_MODEL_NAME = (
    "bangladesh-electricity-demand-bridge"
)

RIDGE_ALPHA = 0.01

MAX_VALIDATED_HORIZON = 8

BACKTEST_OVERALL_MAPE = 7.01


FEATURE_COLUMNS = [
    "lag_1_day",
    "lag_2_day",
    "lag_3_day",
    "lag_7_day",
    "lag_14_day",
    "lag_21_day",
    "lag_30_day",
    "rolling_7_day_mean",
    "rolling_14_day_mean",
    "rolling_30_day_mean",
    "day_of_week",
    "month",
    "day_of_month",
    "day_of_year",
    "is_weekend",
    "is_holiday",
    "trend_days",
]


# ============================================================
# MLflow
# ============================================================

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
        f"MLflow experiment: "
        f"{EXPERIMENT_NAME}"
    )


# ============================================================
# Load National Demand
# ============================================================

def load_national_data():
    logger.info(
        "Loading processed BPDB dataset"
    )

    df = pd.read_csv(
        INPUT_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    national = (
        df.groupby(
            "Date",
            as_index=False,
        )
        .agg(
            total_demand=(
                "Demand (MW)",
                "sum",
            ),
            imputed_rows=(
                "is_imputed",
                "sum",
            ),
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    national["is_real"] = (
        national["imputed_rows"] == 0
    ).astype(int)

    logger.info(
        f"National rows: {len(national)}"
    )

    logger.info(
        "National date range: "
        f"{national['Date'].min().date()} "
        "to "
        f"{national['Date'].max().date()}"
    )

    logger.info(
        f"Real observation days: "
        f"{national['is_real'].sum()}"
    )

    return national


# ============================================================
# Holiday Data
# ============================================================

def build_holiday_set(
    min_date,
    max_date,
):
    bd_holidays = (
        holidays.country_holidays(
            "BD",
            years=range(
                min_date.year,
                max_date.year + 1,
            ),
        )
    )

    return set(
        bd_holidays.keys()
    )


# ============================================================
# Feature Engineering
# ============================================================

def create_feature_row(
    demand_series,
    target_date,
    holiday_dates,
    minimum_date,
):
    values = {}

    # --------------------------------------------------------
    # Exact lag features
    # --------------------------------------------------------

    lag_days = [
        1,
        2,
        3,
        7,
        14,
        21,
        30,
    ]

    for lag in lag_days:
        lag_date = (
            target_date
            - pd.Timedelta(
                days=lag
            )
        )

        if (
            lag_date
            not in demand_series.index
        ):
            return None

        lag_value = (
            demand_series.loc[
                lag_date
            ]
        )

        if pd.isna(
            lag_value
        ):
            return None

        values[
            f"lag_{lag}_day"
        ] = float(
            lag_value
        )

    # --------------------------------------------------------
    # Rolling demand features
    # --------------------------------------------------------

    for window in [
        7,
        14,
        30,
    ]:
        start_date = (
            target_date
            - pd.Timedelta(
                days=window
            )
        )

        end_date = (
            target_date
            - pd.Timedelta(
                days=1
            )
        )

        expected_dates = (
            pd.date_range(
                start=start_date,
                end=end_date,
                freq="D",
            )
        )

        if not expected_dates.isin(
            demand_series.index
        ).all():
            return None

        rolling_values = (
            demand_series.reindex(
                expected_dates
            )
        )

        if rolling_values.isna().any():
            return None

        values[
            f"rolling_{window}_day_mean"
        ] = float(
            rolling_values.mean()
        )

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    values[
        "day_of_week"
    ] = target_date.dayofweek

    values[
        "month"
    ] = target_date.month

    values[
        "day_of_month"
    ] = target_date.day

    values[
        "day_of_year"
    ] = target_date.dayofyear

    # Bangladesh weekend:
    # Friday + Saturday
    values[
        "is_weekend"
    ] = int(
        target_date.dayofweek
        in [4, 5]
    )

    values[
        "is_holiday"
    ] = int(
        target_date.date()
        in holiday_dates
    )

    values[
        "trend_days"
    ] = (
        target_date
        - minimum_date
    ).days

    return values


# ============================================================
# Training Dataset
# ============================================================

def build_training_dataset(
    national,
):
    logger.info(
        "Building Bridge Forecast "
        "training dataset"
    )

    latest_real_date = (
        national.loc[
            national["is_real"] == 1,
            "Date",
        ]
        .max()
    )

    logger.info(
        "Latest real BPDB observation: "
        f"{latest_real_date.date()}"
    )

    training_source = national[
        national["Date"]
        <= latest_real_date
    ].copy()

    demand_series = (
        training_source
        .set_index("Date")[
            "total_demand"
        ]
        .sort_index()
    )

    minimum_date = (
        national["Date"].min()
    )

    holiday_dates = (
        build_holiday_set(
            minimum_date,
            latest_real_date
            + pd.Timedelta(
                days=MAX_VALIDATED_HORIZON
            ),
        )
    )

    rows = []

    for _, row in (
        training_source.iterrows()
    ):
        target_date = row[
            "Date"
        ]

        # Target itself must be a real
        # published BPDB observation.
        if row["is_real"] != 1:
            continue

        feature_row = (
            create_feature_row(
                demand_series=
                    demand_series,
                target_date=
                    target_date,
                holiday_dates=
                    holiday_dates,
                minimum_date=
                    minimum_date,
            )
        )

        if feature_row is None:
            continue

        feature_row[
            "target_demand"
        ] = float(
            row[
                "total_demand"
            ]
        )

        feature_row[
            "Date"
        ] = target_date

        rows.append(
            feature_row
        )

    training_df = pd.DataFrame(
        rows
    )

    if training_df.empty:
        raise ValueError(
            "Bridge training dataset "
            "is empty"
        )

    logger.info(
        f"Bridge training rows: "
        f"{len(training_df)}"
    )

    logger.info(
        "Bridge training range: "
        f"{training_df['Date'].min().date()} "
        "to "
        f"{training_df['Date'].max().date()}"
    )

    return (
        training_df,
        latest_real_date,
    )


# ============================================================
# Model
# ============================================================

def build_model():
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(
                    alpha=RIDGE_ALPHA
                ),
            ),
        ]
    )


# ============================================================
# Save Artifacts
# ============================================================

def save_local_artifacts(
    model,
    training_df,
    latest_real_date,
):
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    logger.info(
        f"Bridge model saved: "
        f"{MODEL_PATH}"
    )

    with open(
        FEATURE_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        for feature in (
            FEATURE_COLUMNS
        ):
            file.write(
                f"{feature}\n"
            )

    training_info = pd.DataFrame(
        [
            {
                "training_rows":
                    len(training_df),

                "training_start":
                    training_df[
                        "Date"
                    ]
                    .min()
                    .date(),

                "training_end":
                    training_df[
                        "Date"
                    ]
                    .max()
                    .date(),

                "latest_real_bpdb_date":
                    latest_real_date.date(),

                "model_type":
                    "Ridge",

                "ridge_alpha":
                    RIDGE_ALPHA,

                "validated_max_horizon_days":
                    MAX_VALIDATED_HORIZON,

                "backtest_overall_mape":
                    BACKTEST_OVERALL_MAPE,
            }
        ]
    )

    training_info.to_csv(
        TRAINING_INFO_PATH,
        index=False,
    )

    logger.info(
        "Bridge training metadata "
        f"saved: {TRAINING_INFO_PATH}"
    )


# ============================================================
# MLflow Registration
# ============================================================

def log_and_register_model(
    model,
    X_train,
    training_df,
    latest_real_date,
):
    logger.info(
        "Logging Bridge Forecast model "
        "to MLflow"
    )

    with mlflow.start_run(
        run_name=(
            "final_bridge_forecast_ridge"
        )
    ):
        mlflow.log_param(
            "model_role",
            "bridge_forecast",
        )

        mlflow.log_param(
            "model_type",
            "Ridge",
        )

        mlflow.log_param(
            "ridge_alpha",
            RIDGE_ALPHA,
        )

        mlflow.log_param(
            "feature_count",
            len(
                FEATURE_COLUMNS
            ),
        )

        mlflow.log_param(
            "training_rows",
            len(
                training_df
            ),
        )

        mlflow.log_param(
            "training_start",
            str(
                training_df[
                    "Date"
                ]
                .min()
                .date()
            ),
        )

        mlflow.log_param(
            "training_end",
            str(
                training_df[
                    "Date"
                ]
                .max()
                .date()
            ),
        )

        mlflow.log_param(
            "latest_real_bpdb_date",
            str(
                latest_real_date.date()
            ),
        )

        mlflow.log_param(
            "validated_max_horizon_days",
            MAX_VALIDATED_HORIZON,
        )

        mlflow.log_metric(
            "backtest_overall_mape",
            BACKTEST_OVERALL_MAPE,
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=(
                X_train.head(5)
            ),
            registered_model_name=(
                REGISTERED_MODEL_NAME
            ),
        )

    logger.info(
        "Bridge Forecast model "
        "registered successfully"
    )


# ============================================================
# Main
# ============================================================

def main():
    logger.info(
        "Starting final Bridge Forecast "
        "model training"
    )

    setup_mlflow()

    national = (
        load_national_data()
    )

    (
        training_df,
        latest_real_date,
    ) = build_training_dataset(
        national
    )

    X_train = training_df[
        FEATURE_COLUMNS
    ]

    y_train = training_df[
        "target_demand"
    ]

    model = build_model()

    logger.info(
        "Training final Bridge Ridge model"
    )

    model.fit(
        X_train,
        y_train,
    )

    save_local_artifacts(
        model=model,
        training_df=training_df,
        latest_real_date=
            latest_real_date,
    )

    log_and_register_model(
        model=model,
        X_train=X_train,
        training_df=training_df,
        latest_real_date=
            latest_real_date,
    )

    print(
        "\nBRIDGE MODEL TRAINING STATUS\n"
    )

    print(
        "Registered model:",
        REGISTERED_MODEL_NAME,
    )

    print(
        "Training rows:",
        len(training_df),
    )

    print(
        "Training range:",
        training_df[
            "Date"
        ]
        .min()
        .date(),
        "to",
        training_df[
            "Date"
        ]
        .max()
        .date(),
    )

    print(
        "Latest real BPDB date:",
        latest_real_date.date(),
    )

    print(
        "Validated maximum horizon:",
        f"{MAX_VALIDATED_HORIZON} days",
    )

    print(
        "Historical backtest MAPE:",
        f"{BACKTEST_OVERALL_MAPE:.2f}%",
    )

    logger.info(
        "Final Bridge Forecast training "
        "completed successfully"
    )


if __name__ == "__main__":
    main()