import os
from pathlib import Path

import holidays
import mlflow
import mlflow.sklearn
import pandas as pd

from dotenv import load_dotenv

from src.logger import get_logger


logger = get_logger(
    "generate_regional_bridge_forecast"
)


# ============================================================
# Configuration
# ============================================================

PROCESSED_DATA_PATH = (
    "data/processed/area_wise_demand_processed.csv"
)

REGIONAL_INFERENCE_PATH = (
    "data/processed/regional_inference_features.csv"
)

OUTPUT_PATH = Path(
    "data/processed/regional_bridge_forecast.csv"
)

MODEL_ALIAS = "champion"
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


ANCHOR_STRATEGIES = {
    "Barisal": "baseline",
    "Chittagong": "ridge",
    "Comilla": "ridge",
    "Dhaka": "ridge",
    "Khulna": "ridge",
    "Mymensingh": "ridge",
    "Rajshahi": "baseline",
    "Rangpur": "ridge",
    "Sylhet": "ridge",
}


ANCHOR_MODEL_NAMES = {
    "Chittagong":
        "bangladesh-electricity-demand-chittagong-ridge",

    "Comilla":
        "bangladesh-electricity-demand-comilla-ridge",

    "Dhaka":
        "bangladesh-electricity-demand-dhaka-ridge",

    "Khulna":
        "bangladesh-electricity-demand-khulna-ridge",

    "Mymensingh":
        "bangladesh-electricity-demand-mymensingh-ridge",

    "Rangpur":
        "bangladesh-electricity-demand-rangpur-ridge",

    "Sylhet":
        "bangladesh-electricity-demand-sylhet-ridge",
}


BRIDGE_MODEL_NAMES = {
    region:
        "bangladesh-electricity-demand-"
        f"{region.lower()}-bridge"
    for region in REGIONS
}


ANCHOR_FEATURE_COLUMNS = [
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


def load_model(
    model_name,
):
    model_uri = (
        f"models:/{model_name}"
        f"@{MODEL_ALIAS}"
    )

    logger.info(
        f"Loading model: {model_uri}"
    )

    return mlflow.sklearn.load_model(
        model_uri
    )


# ============================================================
# Data
# ============================================================

def load_processed_data():
    df = pd.read_csv(
        PROCESSED_DATA_PATH
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

    return (
        df
        .sort_values(
            [
                "region",
                "Date",
            ]
        )
        .reset_index(drop=True)
    )


def load_inference_data():
    df = pd.read_csv(
        REGIONAL_INFERENCE_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"]
    )

    return (
        df
        .sort_values(
            [
                "region",
                "Date",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# Holidays
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

    for lag in [
        1,
        2,
        3,
        7,
        14,
        21,
        30,
    ]:
        lag_date = (
            target_date
            - pd.Timedelta(days=lag)
        )

        if lag_date not in demand_series.index:
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
        expected_dates = pd.date_range(
            start=(
                target_date
                - pd.Timedelta(days=window)
            ),
            end=(
                target_date
                - pd.Timedelta(days=1)
            ),
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
# Anchor Prediction
# ============================================================

def generate_day1_anchor(
    region,
    latest_real_date,
    latest_real_demand,
    region_inference_df,
):
    strategy = (
        ANCHOR_STRATEGIES[
            region
        ]
    )

    forecast_date = (
        latest_real_date
        + pd.Timedelta(days=1)
    )

    if strategy == "baseline":
        return {
            "forecast_date":
                forecast_date,

            "horizon_day":
                1,

            "forecast_mode":
                "anchor",

            "anchor_strategy":
                "baseline",

            "predicted_demand_mw":
                float(
                    latest_real_demand
                ),

            "model_name":
                "persistence_baseline",
        }

    row = region_inference_df[
        (
            region_inference_df[
                "Date"
            ]
            == latest_real_date
        )
        &
        (
            region_inference_df[
                "forecast_date"
            ]
            == forecast_date
        )
    ]

    if row.empty:
        raise ValueError(
            f"No Anchor inference row "
            f"for {region} at "
            f"{latest_real_date.date()}"
        )

    model_name = (
        ANCHOR_MODEL_NAMES[
            region
        ]
    )

    model = load_model(
        model_name
    )

    X = row[
        ANCHOR_FEATURE_COLUMNS
    ]

    prediction = float(
        model.predict(X)[0]
    )

    return {
        "forecast_date":
            forecast_date,

        "horizon_day":
            1,

        "forecast_mode":
            "anchor",

        "anchor_strategy":
            "ridge",

        "predicted_demand_mw":
            prediction,

        "model_name":
            model_name,
    }


# ============================================================
# Regional Forecast
# ============================================================

def generate_region_forecast(
    region,
    processed_df,
    inference_df,
):
    region_df = (
        processed_df[
            processed_df[
                "region"
            ]
            == region
        ]
        .copy()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    region_inference_df = (
        inference_df[
            inference_df[
                "region"
            ]
            == region
        ]
        .copy()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    real_rows = region_df[
        region_df["is_real"] == 1
    ]

    if real_rows.empty:
        raise ValueError(
            f"No real data found for "
            f"{region}"
        )

    latest_real_date = (
        real_rows["Date"].max()
    )

    latest_real_demand = float(
        real_rows.loc[
            real_rows["Date"]
            == latest_real_date,
            "regional_demand_mw",
        ]
        .iloc[0]
    )

    logger.info(
        f"{region} latest real date: "
        f"{latest_real_date.date()}"
    )

    # --------------------------------------------------------
    # Day +1 Anchor
    # --------------------------------------------------------

    anchor_result = (
        generate_day1_anchor(
            region=
                region,

            latest_real_date=
                latest_real_date,

            latest_real_demand=
                latest_real_demand,

            region_inference_df=
                region_inference_df,
        )
    )

    forecasts = [
        anchor_result
    ]

    # --------------------------------------------------------
    # Working demand series
    # --------------------------------------------------------

    historical = region_df[
        region_df["Date"]
        <= latest_real_date
    ].copy()

    demand_series = (
        historical
        .set_index("Date")[
            "regional_demand_mw"
        ]
        .sort_index()
        .copy()
    )

    day1_date = (
        anchor_result[
            "forecast_date"
        ]
    )

    demand_series.loc[
        day1_date
    ] = (
        anchor_result[
            "predicted_demand_mw"
        ]
    )

    # --------------------------------------------------------
    # Bridge model
    # --------------------------------------------------------

    bridge_model_name = (
        BRIDGE_MODEL_NAMES[
            region
        ]
    )

    bridge_model = load_model(
        bridge_model_name
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

    # --------------------------------------------------------
    # Day +2 ... Day +8
    # --------------------------------------------------------

    for horizon_day in range(
        2,
        MAX_VALIDATED_HORIZON + 1,
    ):
        forecast_date = (
            latest_real_date
            + pd.Timedelta(
                days=horizon_day
            )
        )

        feature_row = (
            create_bridge_feature_row(
                demand_series=
                    demand_series,

                target_date=
                    forecast_date,

                holiday_dates=
                    holiday_dates,

                minimum_date=
                    minimum_date,
            )
        )

        if feature_row is None:
            raise ValueError(
                f"Unable to create Bridge "
                f"features for {region} "
                f"{forecast_date.date()}"
            )

        X = pd.DataFrame(
            [feature_row]
        )[BRIDGE_FEATURE_COLUMNS]

        prediction = float(
            bridge_model.predict(
                X
            )[0]
        )

        forecasts.append(
            {
                "forecast_date":
                    forecast_date,

                "horizon_day":
                    horizon_day,

                "forecast_mode":
                    "bridge",

                "anchor_strategy":
                    ANCHOR_STRATEGIES[
                        region
                    ],

                "predicted_demand_mw":
                    prediction,

                "model_name":
                    bridge_model_name,
            }
        )

        demand_series.loc[
            forecast_date
        ] = prediction

    # --------------------------------------------------------
    # Final rows
    # --------------------------------------------------------

    rows = []

    for forecast in forecasts:
        rows.append(
            {
                "region":
                    region,

                "latest_real_bpdb_date":
                    latest_real_date,

                "latest_real_demand_mw":
                    latest_real_demand,

                "forecast_date":
                    forecast[
                        "forecast_date"
                    ],

                "horizon_day":
                    forecast[
                        "horizon_day"
                    ],

                "forecast_mode":
                    forecast[
                        "forecast_mode"
                    ],

                "anchor_strategy":
                    forecast[
                        "anchor_strategy"
                    ],

                "predicted_demand_mw":
                    round(
                        forecast[
                            "predicted_demand_mw"
                        ],
                        2,
                    ),

                "actual_demand_mw":
                    None,

                "actual_available":
                    False,

                "forecast_status":
                    (
                        "real_data_anchored"
                        if forecast[
                            "horizon_day"
                        ]
                        == 1
                        else
                        "recursive_estimate"
                    ),

                "model_name":
                    forecast[
                        "model_name"
                    ],

                "model_alias":
                    (
                        MODEL_ALIAS
                        if forecast[
                            "model_name"
                        ]
                        != "persistence_baseline"
                        else None
                    ),

                "validated_horizon":
                    True,
            }
        )

    return rows


# ============================================================
# Main
# ============================================================

def main():
    logger.info(
        "Starting Regional Extended "
        "Forecast generation"
    )

    setup_mlflow()

    processed_df = (
        load_processed_data()
    )

    inference_df = (
        load_inference_data()
    )

    all_forecasts = []

    for region in REGIONS:
        logger.info(
            "=" * 60
        )

        logger.info(
            f"Generating forecast: "
            f"{region}"
        )

        regional_rows = (
            generate_region_forecast(
                region=
                    region,

                processed_df=
                    processed_df,

                inference_df=
                    inference_df,
            )
        )

        all_forecasts.extend(
            regional_rows
        )

    output_df = pd.DataFrame(
        all_forecasts
    )

    output_df[
        "forecast_date"
    ] = pd.to_datetime(
        output_df[
            "forecast_date"
        ]
    )

    output_df[
        "latest_real_bpdb_date"
    ] = pd.to_datetime(
        output_df[
            "latest_real_bpdb_date"
        ]
    )

    output_df = (
        output_df
        .sort_values(
            [
                "region",
                "horizon_day",
            ]
        )
        .reset_index(drop=True)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nREGIONAL EXTENDED "
        "FORECAST SUMMARY\n"
    )

    summary = (
        output_df.groupby(
            "region"
        )
        .agg(
            latest_real_date=(
                "latest_real_bpdb_date",
                "first",
            ),
            forecast_start=(
                "forecast_date",
                "min",
            ),
            forecast_end=(
                "forecast_date",
                "max",
            ),
            forecast_count=(
                "forecast_date",
                "count",
            ),
        )
        .reset_index()
    )

    summary[
        "latest_real_date"
    ] = (
        summary[
            "latest_real_date"
        ]
        .dt.date
    )

    summary[
        "forecast_start"
    ] = (
        summary[
            "forecast_start"
        ]
        .dt.date
    )

    summary[
        "forecast_end"
    ] = (
        summary[
            "forecast_end"
        ]
        .dt.date
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nSAMPLE: DHAKA\n"
    )

    print(
        output_df[
            output_df[
                "region"
            ]
            == "Dhaka"
        ][
            [
                "forecast_date",
                "horizon_day",
                "forecast_mode",
                "anchor_strategy",
                "predicted_demand_mw",
                "forecast_status",
            ]
        ].to_string(
            index=False
        )
    )

    logger.info(
        "Regional Extended Forecast "
        "generation completed successfully"
    )


if __name__ == "__main__":
    main()