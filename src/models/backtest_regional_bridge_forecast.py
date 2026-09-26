from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger


logger = get_logger("backtest_regional_bridge_forecast")


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = (
    "data/processed/area_wise_demand_processed.csv"
)

PREDICTIONS_PATH = Path(
    "artifacts/regional_bridge_backtest_predictions.csv"
)

SUMMARY_PATH = Path(
    "artifacts/regional_bridge_backtest_summary.csv"
)

COMPARISON_PATH = Path(
    "artifacts/regional_bridge_model_comparison.csv"
)


FORECAST_HORIZON = 8

RIDGE_ALPHA = 0.01

BACKTEST_START = pd.Timestamp(
    "2025-01-01"
)

BACKTEST_STEP_DAYS = 7


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
# Metrics
# ============================================================

def calculate_mape(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    mask = y_true != 0

    if not mask.any():
        return np.nan

    return (
        np.mean(
            np.abs(
                (
                    y_true[mask]
                    - y_pred[mask]
                )
                / y_true[mask]
            )
        )
        * 100
    )


# ============================================================
# Load Regional Demand
# ============================================================

def load_regional_data():
    logger.info(
        "Loading processed regional BPDB dataset"
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

    logger.info(
        "Date range: "
        f"{df['Date'].min().date()} "
        "to "
        f"{df['Date'].max().date()}"
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
# Features
# ============================================================

def create_feature_row(
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
    training_end_date,
    holiday_dates,
):
    historical = region_df[
        region_df["Date"]
        <= training_end_date
    ].copy()

    demand_series = (
        historical
        .set_index("Date")[
            "regional_demand_mw"
        ]
        .sort_index()
    )

    minimum_date = (
        region_df["Date"].min()
    )

    rows = []

    for _, row in (
        historical.iterrows()
    ):
        target_date = row[
            "Date"
        ]

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
                "regional_demand_mw"
            ]
        )

        feature_row[
            "Date"
        ] = target_date

        rows.append(
            feature_row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Model
# ============================================================

def create_model():
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
# Recursive Forecast
# ============================================================

def recursive_forecast(
    model,
    region_df,
    anchor_date,
    holiday_dates,
):
    historical = region_df[
        region_df["Date"]
        <= anchor_date
    ].copy()

    working_series = (
        historical
        .set_index("Date")[
            "regional_demand_mw"
        ]
        .sort_index()
        .copy()
    )

    minimum_date = (
        region_df["Date"].min()
    )

    predictions = []

    for horizon_day in range(
        1,
        FORECAST_HORIZON + 1,
    ):
        forecast_date = (
            anchor_date
            + pd.Timedelta(
                days=horizon_day
            )
        )

        feature_row = (
            create_feature_row(
                demand_series=
                    working_series,

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
                "Unable to create "
                "regional bridge features "
                f"for {forecast_date.date()}"
            )

        X = pd.DataFrame(
            [feature_row]
        )[FEATURE_COLUMNS]

        prediction = float(
            model.predict(
                X
            )[0]
        )

        predictions.append(
            {
                "horizon_day":
                    horizon_day,

                "forecast_date":
                    forecast_date,

                "prediction":
                    prediction,
            }
        )

        working_series.loc[
            forecast_date
        ] = prediction

    return predictions


# ============================================================
# Backtest Origins
# ============================================================

def get_backtest_origins(
    region_df,
):
    latest_real_date = (
        region_df.loc[
            region_df["is_real"] == 1,
            "Date",
        ]
        .max()
    )

    latest_possible_origin = (
        latest_real_date
        - pd.Timedelta(
            days=FORECAST_HORIZON
        )
    )

    origins = pd.date_range(
        start=BACKTEST_START,
        end=latest_possible_origin,
        freq=f"{BACKTEST_STEP_DAYS}D",
    )

    lookup = (
        region_df
        .set_index("Date")
    )

    valid_origins = []

    for origin in origins:
        if origin not in lookup.index:
            continue

        if (
            lookup.loc[
                origin,
                "is_real",
            ]
            != 1
        ):
            continue

        future_dates = pd.date_range(
            start=(
                origin
                + pd.Timedelta(
                    days=1
                )
            ),
            periods=FORECAST_HORIZON,
            freq="D",
        )

        if not future_dates.isin(
            lookup.index
        ).all():
            continue

        future_rows = (
            lookup.loc[
                future_dates
            ]
        )

        if not (
            future_rows[
                "is_real"
            ]
            == 1
        ).all():
            continue

        valid_origins.append(
            origin
        )

    return valid_origins


# ============================================================
# Region Backtest
# ============================================================

def backtest_region(
    region_df,
    region,
):
    logger.info(
        f"Starting Bridge backtest for: "
        f"{region}"
    )

    minimum_date = (
        region_df["Date"].min()
    )

    maximum_date = (
        region_df["Date"].max()
        + pd.Timedelta(
            days=FORECAST_HORIZON
        )
    )

    holiday_dates = (
        build_holiday_set(
            minimum_date,
            maximum_date,
        )
    )

    origins = get_backtest_origins(
        region_df
    )

    logger.info(
        f"{region} valid backtest origins: "
        f"{len(origins)}"
    )

    lookup = (
        region_df
        .set_index("Date")
    )

    results = []

    for index, anchor_date in enumerate(
        origins,
        start=1,
    ):
        logger.info(
            f"{region} | "
            f"{index}/{len(origins)} | "
            f"Anchor: {anchor_date.date()}"
        )

        training_df = (
            build_training_dataset(
                region_df=
                    region_df,

                training_end_date=
                    anchor_date,

                holiday_dates=
                    holiday_dates,
            )
        )

        if training_df.empty:
            continue

        X_train = training_df[
            FEATURE_COLUMNS
        ]

        y_train = training_df[
            "target_demand"
        ]

        model = create_model()

        model.fit(
            X_train,
            y_train,
        )

        forecasts = (
            recursive_forecast(
                model=
                    model,

                region_df=
                    region_df,

                anchor_date=
                    anchor_date,

                holiday_dates=
                    holiday_dates,
            )
        )

        anchor_demand = float(
            lookup.loc[
                anchor_date,
                "regional_demand_mw",
            ]
        )

        for forecast in forecasts:
            forecast_date = (
                forecast[
                    "forecast_date"
                ]
            )

            actual = float(
                lookup.loc[
                    forecast_date,
                    "regional_demand_mw",
                ]
            )

            bridge_prediction = float(
                forecast[
                    "prediction"
                ]
            )

            baseline_prediction = (
                anchor_demand
            )

            results.append(
                {
                    "region":
                        region,

                    "anchor_date":
                        anchor_date,

                    "forecast_date":
                        forecast_date,

                    "horizon_day":
                        forecast[
                            "horizon_day"
                        ],

                    "actual_demand_mw":
                        actual,

                    "bridge_prediction_mw":
                        bridge_prediction,

                    "baseline_prediction_mw":
                        baseline_prediction,

                    "bridge_absolute_error_mw":
                        abs(
                            bridge_prediction
                            - actual
                        ),

                    "baseline_absolute_error_mw":
                        abs(
                            baseline_prediction
                            - actual
                        ),

                    "bridge_percentage_error":
                        (
                            abs(
                                bridge_prediction
                                - actual
                            )
                            / actual
                            * 100
                        ),

                    "baseline_percentage_error":
                        (
                            abs(
                                baseline_prediction
                                - actual
                            )
                            / actual
                            * 100
                        ),
                }
            )

    return pd.DataFrame(
        results
    )


# ============================================================
# Summary
# ============================================================

def build_summary(
    predictions,
):
    rows = []

    grouped = predictions.groupby(
        [
            "region",
            "horizon_day",
        ]
    )

    for (
        region,
        horizon_day,
    ), group in grouped:
        actual = group[
            "actual_demand_mw"
        ]

        bridge = group[
            "bridge_prediction_mw"
        ]

        baseline = group[
            "baseline_prediction_mw"
        ]

        rows.append(
            {
                "region":
                    region,

                "horizon_day":
                    horizon_day,

                "samples":
                    len(group),

                "bridge_mae":
                    mean_absolute_error(
                        actual,
                        bridge,
                    ),

                "bridge_rmse":
                    (
                        mean_squared_error(
                            actual,
                            bridge,
                        )
                        ** 0.5
                    ),

                "bridge_mape":
                    calculate_mape(
                        actual,
                        bridge,
                    ),

                "baseline_mae":
                    mean_absolute_error(
                        actual,
                        baseline,
                    ),

                "baseline_rmse":
                    (
                        mean_squared_error(
                            actual,
                            baseline,
                        )
                        ** 0.5
                    ),

                "baseline_mape":
                    calculate_mape(
                        actual,
                        baseline,
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Region-Level Comparison
# ============================================================

def build_region_comparison(
    predictions,
):
    rows = []

    for region, group in (
        predictions.groupby(
            "region"
        )
    ):
        bridge_mape = (
            calculate_mape(
                group[
                    "actual_demand_mw"
                ],
                group[
                    "bridge_prediction_mw"
                ],
            )
        )

        baseline_mape = (
            calculate_mape(
                group[
                    "actual_demand_mw"
                ],
                group[
                    "baseline_prediction_mw"
                ],
            )
        )

        rows.append(
            {
                "region":
                    region,

                "bridge_mape":
                    bridge_mape,

                "baseline_mape":
                    baseline_mape,

                "bridge_improvement_mape_points":
                    (
                        baseline_mape
                        - bridge_mape
                    ),

                "recommended_strategy":
                    (
                        "bridge"
                        if bridge_mape
                        < baseline_mape
                        else "baseline"
                    ),
            }
        )

    comparison = pd.DataFrame(
        rows
    )

    comparison = (
        comparison
        .sort_values(
            "bridge_improvement_mape_points",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return comparison


# ============================================================
# Overall Summary
# ============================================================

def print_overall_summary(
    predictions,
):
    bridge_mape = (
        calculate_mape(
            predictions[
                "actual_demand_mw"
            ],
            predictions[
                "bridge_prediction_mw"
            ],
        )
    )

    baseline_mape = (
        calculate_mape(
            predictions[
                "actual_demand_mw"
            ],
            predictions[
                "baseline_prediction_mw"
            ],
        )
    )

    bridge_mae = (
        mean_absolute_error(
            predictions[
                "actual_demand_mw"
            ],
            predictions[
                "bridge_prediction_mw"
            ],
        )
    )

    baseline_mae = (
        mean_absolute_error(
            predictions[
                "actual_demand_mw"
            ],
            predictions[
                "baseline_prediction_mw"
            ],
        )
    )

    print(
        "\nOVERALL REGIONAL BRIDGE SUMMARY\n"
    )

    print(
        f"Bridge MAE:    "
        f"{bridge_mae:.2f} MW"
    )

    print(
        f"Bridge MAPE:   "
        f"{bridge_mape:.2f}%"
    )

    print(
        f"Baseline MAE:  "
        f"{baseline_mae:.2f} MW"
    )

    print(
        f"Baseline MAPE: "
        f"{baseline_mape:.2f}%"
    )


# ============================================================
# Main
# ============================================================

def main():
    logger.info(
        "Starting Regional Bridge "
        "Forecast backtest"
    )

    df = load_regional_data()

    all_results = []

    regions = sorted(
        df["region"].unique()
    )

    for region in regions:
        region_df = (
            df[
                df["region"] == region
            ]
            .copy()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        result = backtest_region(
            region_df=
                region_df,

            region=
                region,
        )

        if not result.empty:
            all_results.append(
                result
            )

    if not all_results:
        raise ValueError(
            "Regional Bridge backtest "
            "produced no results"
        )

    predictions = pd.concat(
        all_results,
        ignore_index=True,
    )

    summary = build_summary(
        predictions
    )

    comparison = (
        build_region_comparison(
            predictions
        )
    )

    PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    print(
        "\nREGIONAL BRIDGE MODEL COMPARISON\n"
    )

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print(
        "\nREGIONAL BRIDGE HORIZON SUMMARY\n"
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print_overall_summary(
        predictions
    )

    logger.info(
        "Regional Bridge backtest "
        "completed successfully"
    )


if __name__ == "__main__":
    main()