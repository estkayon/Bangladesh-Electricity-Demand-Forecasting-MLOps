import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import holidays
import mlflow
import mlflow.sklearn
import pandas as pd

from dotenv import load_dotenv

from src.logger import get_logger


logger = get_logger("generate_bridge_forecast")


# ============================================================
# Configuration
# ============================================================

PROCESSED_DATA_PATH = (
    "data/processed/area_wise_demand_processed.csv"
)

NATIONAL_INFERENCE_PATH = (
    "data/processed/national_inference_features.csv"
)

OUTPUT_PATH = Path(
    "data/processed/bridge_forecast.csv"
)


ANCHOR_MODEL_NAME = (
    "bangladesh-electricity-demand-ridge"
)

BRIDGE_MODEL_NAME = (
    "bangladesh-electricity-demand-bridge"
)

MODEL_ALIAS = "champion"

MAX_VALIDATED_HORIZON = 8

BANGLADESH_TIMEZONE = ZoneInfo(
    "Asia/Dhaka"
)


ANCHOR_FEATURE_COLUMNS = [
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

    logger.info(
        "MLflow tracking configured"
    )


def load_models():
    anchor_uri = (
        f"models:/{ANCHOR_MODEL_NAME}"
        f"@{MODEL_ALIAS}"
    )

    bridge_uri = (
        f"models:/{BRIDGE_MODEL_NAME}"
        f"@{MODEL_ALIAS}"
    )

    logger.info(
        f"Loading Anchor model: "
        f"{anchor_uri}"
    )

    anchor_model = (
        mlflow.sklearn.load_model(
            anchor_uri
        )
    )

    logger.info(
        f"Loading Bridge model: "
        f"{bridge_uri}"
    )

    bridge_model = (
        mlflow.sklearn.load_model(
            bridge_uri
        )
    )

    return (
        anchor_model,
        bridge_model,
    )


# ============================================================
# Load Demand Data
# ============================================================

def load_national_demand():
    logger.info(
        "Loading processed BPDB data"
    )

    df = pd.read_csv(
        PROCESSED_DATA_PATH
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
        national[
            "imputed_rows"
        ]
        == 0
    ).astype(int)

    return national


def get_latest_real_date(
    national,
):
    real_rows = national[
        national["is_real"] == 1
    ]

    if real_rows.empty:
        raise ValueError(
            "No real BPDB observations found"
        )

    latest_real_date = (
        real_rows[
            "Date"
        ]
        .max()
    )

    logger.info(
        "Latest real BPDB date: "
        f"{latest_real_date.date()}"
    )

    return latest_real_date


# ============================================================
# Anchor Forecast
# ============================================================

def load_latest_anchor_features(
    latest_real_date,
):
    logger.info(
        "Loading latest Anchor "
        "inference features"
    )

    df = pd.read_csv(
        NATIONAL_INFERENCE_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df["forecast_date"] = (
        pd.to_datetime(
            df["forecast_date"]
        )
    )

    selected = df[
        df["Date"]
        == latest_real_date
    ]

    if selected.empty:
        raise ValueError(
            "No Anchor inference row found "
            f"for {latest_real_date.date()}. "
            "Run build_inference_features first."
        )

    return selected.iloc[-1]


def generate_anchor_prediction(
    anchor_model,
    row,
):
    X = pd.DataFrame(
        [
            row[
                ANCHOR_FEATURE_COLUMNS
            ].to_dict()
        ]
    )

    prediction = float(
        anchor_model.predict(
            X
        )[0]
    )

    forecast_date = (
        pd.Timestamp(
            row["forecast_date"]
        )
    )

    logger.info(
        "Anchor Forecast: "
        f"{forecast_date.date()} "
        f"-> {prediction:.2f} MW"
    )

    return {
        "forecast_date":
            forecast_date,

        "horizon_day":
            1,

        "forecast_mode":
            "anchor",

        "predicted_demand_mw":
            prediction,

        "model_name":
            ANCHOR_MODEL_NAME,
    }


# ============================================================
# Bridge Features
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


def create_bridge_feature_row(
    demand_series,
    target_date,
    holiday_dates,
    minimum_date,
):
    values = {}

    # --------------------------------------------------------
    # Exact lags
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

        value = demand_series.loc[
            lag_date
        ]

        if pd.isna(value):
            return None

        values[
            f"lag_{lag}_day"
        ] = float(value)

    # --------------------------------------------------------
    # Rolling features
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
    # Calendar
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
# Target Horizon
# ============================================================

def get_target_end_date(
    latest_real_date,
):
    today = datetime.now(
        BANGLADESH_TIMEZONE
    ).date()

    # User-facing goal:
    # keep forecasts available through
    # tomorrow.
    desired_end_date = (
        pd.Timestamp(today)
        + pd.Timedelta(days=1)
    )

    validated_end_date = (
        latest_real_date
        + pd.Timedelta(
            days=MAX_VALIDATED_HORIZON
        )
    )

    target_end_date = min(
        desired_end_date,
        validated_end_date,
    )

    logger.info(
        "Desired forecast end date: "
        f"{desired_end_date.date()}"
    )

    logger.info(
        "Validated maximum end date: "
        f"{validated_end_date.date()}"
    )

    logger.info(
        "Selected forecast end date: "
        f"{target_end_date.date()}"
    )

    return (
        target_end_date,
        desired_end_date,
        validated_end_date,
    )


# ============================================================
# Recursive Bridge Forecast
# ============================================================

def generate_bridge_predictions(
    bridge_model,
    national,
    latest_real_date,
    anchor_prediction,
    target_end_date,
):
    # --------------------------------------------------------
    # Historical working series
    # --------------------------------------------------------

    historical = national[
        national["Date"]
        <= latest_real_date
    ].copy()

    working_series = (
        historical
        .set_index("Date")[
            "total_demand"
        ]
        .sort_index()
        .copy()
    )

    minimum_date = (
        national["Date"].min()
    )

    holiday_dates = (
        build_holiday_set(
            minimum_date,
            target_end_date,
        )
    )

    # --------------------------------------------------------
    # Insert Day+1 Anchor prediction
    # --------------------------------------------------------

    anchor_date = (
        anchor_prediction[
            "forecast_date"
        ]
    )

    working_series.loc[
        anchor_date
    ] = (
        anchor_prediction[
            "predicted_demand_mw"
        ]
    )

    results = [
        anchor_prediction
    ]

    # --------------------------------------------------------
    # Day+2 onwards
    # --------------------------------------------------------

    current_date = (
        anchor_date
        + pd.Timedelta(days=1)
    )

    while (
        current_date
        <= target_end_date
    ):
        horizon_day = (
            current_date
            - latest_real_date
        ).days

        if (
            horizon_day
            > MAX_VALIDATED_HORIZON
        ):
            logger.warning(
                "Maximum validated Bridge "
                "horizon reached"
            )
            break

        feature_row = (
            create_bridge_feature_row(
                demand_series=
                    working_series,
                target_date=
                    current_date,
                holiday_dates=
                    holiday_dates,
                minimum_date=
                    minimum_date,
            )
        )

        if feature_row is None:
            raise ValueError(
                "Unable to build Bridge "
                f"features for "
                f"{current_date.date()}"
            )

        X = pd.DataFrame(
            [feature_row]
        )[BRIDGE_FEATURE_COLUMNS]

        prediction = float(
            bridge_model.predict(
                X
            )[0]
        )

        logger.info(
            "Bridge Forecast "
            f"Day+{horizon_day}: "
            f"{current_date.date()} "
            f"-> {prediction:.2f} MW"
        )

        results.append(
            {
                "forecast_date":
                    current_date,

                "horizon_day":
                    horizon_day,

                "forecast_mode":
                    "bridge",

                "predicted_demand_mw":
                    prediction,

                "model_name":
                    BRIDGE_MODEL_NAME,
            }
        )

        # Recursive step
        working_series.loc[
            current_date
        ] = prediction

        current_date += (
            pd.Timedelta(days=1)
        )

    return results


# ============================================================
# Save Forecast
# ============================================================

def save_forecast(
    predictions,
    latest_real_date,
):
    forecast_df = pd.DataFrame(
        predictions
    )

    forecast_df[
        "latest_real_bpdb_date"
    ] = latest_real_date.date()

    forecast_df[
        "actual_demand_mw"
    ] = None

    forecast_df[
        "actual_available"
    ] = False

    forecast_df[
        "forecast_status"
    ] = forecast_df[
        "forecast_mode"
    ].map(
        {
            "anchor":
                "real_data_anchored",

            "bridge":
                "recursive_estimate",
        }
    )

    forecast_df[
        "model_alias"
    ] = MODEL_ALIAS

    forecast_df[
        "validated_horizon"
    ] = (
        forecast_df[
            "horizon_day"
        ]
        <= MAX_VALIDATED_HORIZON
    )

    forecast_df = (
        forecast_df
        .sort_values(
            "forecast_date"
        )
        .reset_index(drop=True)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast_df.to_csv(
        OUTPUT_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    logger.info(
        f"Forecast saved to: "
        f"{OUTPUT_PATH}"
    )

    return forecast_df


# ============================================================
# Main
# ============================================================

def main():
    logger.info(
        "Starting Anchor + Bridge "
        "forecast generation"
    )

    setup_mlflow()

    (
        anchor_model,
        bridge_model,
    ) = load_models()

    national = (
        load_national_demand()
    )

    latest_real_date = (
        get_latest_real_date(
            national
        )
    )

    anchor_row = (
        load_latest_anchor_features(
            latest_real_date
        )
    )

    anchor_prediction = (
        generate_anchor_prediction(
            anchor_model,
            anchor_row,
        )
    )

    (
        target_end_date,
        desired_end_date,
        validated_end_date,
    ) = get_target_end_date(
        latest_real_date
    )

    predictions = (
        generate_bridge_predictions(
            bridge_model=
                bridge_model,

            national=
                national,

            latest_real_date=
                latest_real_date,

            anchor_prediction=
                anchor_prediction,

            target_end_date=
                target_end_date,
        )
    )

    forecast_df = save_forecast(
        predictions=
            predictions,

        latest_real_date=
            latest_real_date,
    )

    # ========================================================
    # Console Summary
    # ========================================================

    print(
        "\nANCHOR + BRIDGE FORECAST\n"
    )

    print(
        "Latest real BPDB date:",
        latest_real_date.date(),
    )

    print(
        "Validated maximum horizon:",
        f"{MAX_VALIDATED_HORIZON} days",
    )

    print()

    display_columns = [
        "forecast_date",
        "horizon_day",
        "forecast_mode",
        "predicted_demand_mw",
        "forecast_status",
    ]

    print(
        forecast_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.2f}"
            ),
        )
    )

    print()

    if (
        desired_end_date
        > validated_end_date
    ):
        print(
            "WARNING:"
        )

        print(
            "The requested current/tomorrow "
            "forecast extends beyond the "
            "validated Bridge horizon."
        )

        print(
            "Forecasts were capped at:",
            validated_end_date.date(),
        )

    else:
        print(
            "Forecast coverage reaches:",
            target_end_date.date(),
        )

    logger.info(
        "Anchor + Bridge forecast "
        "generation completed successfully"
    )


if __name__ == "__main__":
    main()