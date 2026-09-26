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


logger = get_logger(
    "train_regional_bridge_models"
)


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = (
    "data/processed/area_wise_demand_processed.csv"
)

ARTIFACT_DIR = Path(
    "artifacts/regional_bridge_models"
)

EXPERIMENT_NAME = (
    "Bangladesh-Electricity-Demand-Forecasting"
)

RIDGE_ALPHA = 0.01
MAX_VALIDATED_HORIZON = 8


REGIONS = [
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


BRIDGE_FEATURE_COLUMNS = [
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
# Data
# ============================================================

def load_regional_data():
    logger.info(
        "Loading processed regional dataset"
    )

    df = pd.read_csv(
        INPUT_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.rename(
        columns={
            "Zone Name":
                "region",

            "Demand (MW)":
                "regional_demand_mw",

            "Load shed (MW)":
                "regional_load_shed_mw",
        }
    )

    df["is_real"] = (
        df["is_imputed"] == 0
    ).astype(int)

    df = (
        df
        .sort_values(
            [
                "region",
                "Date",
            ]
        )
        .reset_index(drop=True)
    )

    logger.info(
        f"Rows: {len(df)}"
    )

    logger.info(
        f"Regions: "
        f"{df['region'].nunique()}"
    )

    return df


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
# Bridge Features
# ============================================================

def create_bridge_feature_row(
    demand_series,
    target_date,
    holiday_dates,
    minimum_date,
):
    values = {}

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

        value = demand_series.loc[
            lag_date
        ]

        if pd.isna(value):
            return None

        values[
            f"lag_{lag}_day"
        ] = float(value)

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
            - pd.Timedelta(days=1)
        )

        expected_dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq="D",
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
    region_df,
):
    latest_real_date = (
        region_df.loc[
            region_df["is_real"] == 1,
            "Date",
        ]
        .max()
    )

    logger.info(
        "Latest real date: "
        f"{latest_real_date.date()}"
    )

    training_source = (
        region_df[
            region_df["Date"]
            <= latest_real_date
        ]
        .copy()
    )

    demand_series = (
        training_source
        .set_index("Date")[
            "regional_demand_mw"
        ]
        .sort_index()
    )

    minimum_date = (
        region_df["Date"].min()
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
        if row["is_real"] != 1:
            continue

        target_date = row["Date"]

        feature_row = (
            create_bridge_feature_row(
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
                "regional_demand_mw"
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
# Utilities
# ============================================================

def sanitize_region(
    region,
):
    return (
        region.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


# ============================================================
# Train + Register
# ============================================================

def train_region(
    df,
    region,
):
    logger.info(
        "=" * 60
    )

    logger.info(
        f"Training regional Bridge model: "
        f"{region}"
    )

    region_df = (
        df[
            df["region"] == region
        ]
        .copy()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    (
        training_df,
        latest_real_date,
    ) = build_training_dataset(
        region_df
    )

    X_train = training_df[
        BRIDGE_FEATURE_COLUMNS
    ]

    y_train = training_df[
        "target_demand"
    ]

    model = build_model()

    model.fit(
        X_train,
        y_train,
    )

    safe_region = sanitize_region(
        region
    )

    local_model_path = (
        ARTIFACT_DIR
        / f"{safe_region}_bridge_ridge.pkl"
    )

    joblib.dump(
        model,
        local_model_path,
    )

    logger.info(
        f"Local model saved: "
        f"{local_model_path}"
    )

    registered_model_name = (
        "bangladesh-electricity-demand-"
        f"{safe_region}-bridge"
    )

    run_name = (
        f"final_{safe_region}_bridge"
    )

    with mlflow.start_run(
        run_name=run_name
    ):
        mlflow.log_param(
            "region",
            region,
        )

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
            "training_rows",
            len(training_df),
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
            "feature_count",
            len(
                BRIDGE_FEATURE_COLUMNS
            ),
        )

        mlflow.log_param(
            "validated_max_horizon_days",
            MAX_VALIDATED_HORIZON,
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=(
                X_train.head(5)
            ),
            registered_model_name=(
                registered_model_name
            ),
        )

    logger.info(
        "Registered model: "
        f"{registered_model_name}"
    )

    return {
        "region":
            region,

        "registered_model_name":
            registered_model_name,

        "training_rows":
            len(training_df),

        "training_start":
            str(
                training_df[
                    "Date"
                ]
                .min()
                .date()
            ),

        "training_end":
            str(
                training_df[
                    "Date"
                ]
                .max()
                .date()
            ),

        "latest_real_bpdb_date":
            str(
                latest_real_date.date()
            ),

        "local_model_path":
            str(
                local_model_path
            ),
    }


# ============================================================
# Main
# ============================================================

def main():
    logger.info(
        "Starting Regional Bridge "
        "model training"
    )

    setup_mlflow()

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_regional_data()

    results = []

    for region in REGIONS:
        result = train_region(
            df=df,
            region=region,
        )

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    summary_path = (
        ARTIFACT_DIR
        / "regional_bridge_training_summary.csv"
    )

    results_df.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\nREGIONAL BRIDGE MODEL "
        "TRAINING STATUS\n"
    )

    print(
        results_df[
            [
                "region",
                "registered_model_name",
                "training_rows",
                "training_start",
                "training_end",
                "latest_real_bpdb_date",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Validated maximum horizon:",
        f"{MAX_VALIDATED_HORIZON} days",
    )

    logger.info(
        "Regional Bridge model training "
        "completed successfully"
    )


if __name__ == "__main__":
    main()