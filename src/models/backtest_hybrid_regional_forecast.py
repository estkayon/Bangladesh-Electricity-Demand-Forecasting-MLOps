from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger


logger = get_logger(
    "backtest_hybrid_regional_forecast"
)


# ============================================================
# Configuration
# ============================================================

PROCESSED_DATA_PATH = (
    "data/processed/area_wise_demand_processed.csv"
)

REGIONAL_FEATURE_PATH = (
    "data/processed/regional_model_features.csv"
)

PREDICTIONS_PATH = Path(
    "artifacts/hybrid_regional_backtest_predictions.csv"
)

SUMMARY_PATH = Path(
    "artifacts/hybrid_regional_backtest_summary.csv"
)

COMPARISON_PATH = Path(
    "artifacts/hybrid_regional_model_comparison.csv"
)


FORECAST_HORIZON = 8

RIDGE_ALPHA = 0.01

BACKTEST_START = pd.Timestamp(
    "2025-01-01"
)

BACKTEST_STEP_DAYS = 7


# ============================================================
# Deployed Anchor Strategy
# ============================================================

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


# ============================================================
# Anchor Model Features
# ============================================================

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


# ============================================================
# Bridge Features
# ============================================================

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
# Load Data
# ============================================================

def load_processed_regional_data():
    logger.info(
        "Loading processed regional demand data"
    )

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
        f"Processed rows: {len(df)}"
    )

    return df


def load_anchor_feature_data():
    logger.info(
        "Loading regional Anchor feature data"
    )

    df = pd.read_csv(
        REGIONAL_FEATURE_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df["forecast_date"] = (
        pd.to_datetime(
            df["forecast_date"]
        )
    )

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
        f"Anchor feature rows: {len(df)}"
    )

    return df


# ============================================================
# Models
# ============================================================

def build_ridge_model():
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
# Bridge Feature Creation
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
# Bridge Training Dataset
# ============================================================

def build_bridge_training_dataset(
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
        if row["is_real"] != 1:
            continue

        target_date = row[
            "Date"
        ]

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

    return pd.DataFrame(
        rows
    )


# ============================================================
# Day+1 Anchor Forecast
# ============================================================

def generate_anchor_prediction(
    region,
    anchor_date,
    region_demand_df,
    region_feature_df,
):
    strategy = (
        ANCHOR_STRATEGIES[
            region
        ]
    )

    demand_lookup = (
        region_demand_df
        .set_index("Date")
    )

    anchor_demand = float(
        demand_lookup.loc[
            anchor_date,
            "regional_demand_mw",
        ]
    )

    # --------------------------------------------------------
    # Baseline Anchor
    # --------------------------------------------------------

    if strategy == "baseline":
        return (
            anchor_demand,
            "baseline",
        )

    # --------------------------------------------------------
    # Ridge Anchor
    # --------------------------------------------------------

    forecast_date = (
        anchor_date
        + pd.Timedelta(days=1)
    )

    # Training rows must represent forecasts
    # whose target date is no later than the
    # current anchor date.
    training_df = (
        region_feature_df[
            region_feature_df[
                "forecast_date"
            ]
            <= anchor_date
        ]
        .copy()
    )

    prediction_row = (
        region_feature_df[
            (
                region_feature_df[
                    "Date"
                ]
                == anchor_date
            )
            & (
                region_feature_df[
                    "forecast_date"
                ]
                == forecast_date
            )
        ]
        .copy()
    )

    if training_df.empty:
        raise ValueError(
            f"No Anchor training rows "
            f"for {region} at "
            f"{anchor_date.date()}"
        )

    if prediction_row.empty:
        raise ValueError(
            f"No Anchor prediction feature "
            f"row for {region} at "
            f"{anchor_date.date()}"
        )

    X_train = training_df[
        ANCHOR_FEATURE_COLUMNS
    ]

    y_train = training_df[
        "next_day_demand_mw"
    ]

    model = build_ridge_model()

    model.fit(
        X_train,
        y_train,
    )

    X_predict = prediction_row[
        ANCHOR_FEATURE_COLUMNS
    ]

    prediction = float(
        model.predict(
            X_predict
        )[0]
    )

    return (
        prediction,
        "ridge",
    )


# ============================================================
# Recursive Hybrid Forecast
# ============================================================

def generate_hybrid_forecast(
    region,
    anchor_date,
    region_demand_df,
    region_feature_df,
    bridge_model,
    holiday_dates,
):
    historical = (
        region_demand_df[
            region_demand_df[
                "Date"
            ]
            <= anchor_date
        ]
        .copy()
    )

    working_series = (
        historical
        .set_index("Date")[
            "regional_demand_mw"
        ]
        .sort_index()
        .copy()
    )

    minimum_date = (
        region_demand_df[
            "Date"
        ]
        .min()
    )

    predictions = []

    # --------------------------------------------------------
    # Day +1 = deployed Anchor strategy
    # --------------------------------------------------------

    (
        day1_prediction,
        anchor_strategy,
    ) = generate_anchor_prediction(
        region=region,
        anchor_date=anchor_date,
        region_demand_df=
            region_demand_df,
        region_feature_df=
            region_feature_df,
    )

    day1_date = (
        anchor_date
        + pd.Timedelta(days=1)
    )

    predictions.append(
        {
            "horizon_day": 1,
            "forecast_date":
                day1_date,
            "prediction":
                day1_prediction,
            "forecast_mode":
                "anchor",
            "strategy":
                anchor_strategy,
        }
    )

    # Day+1 predicted value becomes
    # recursive history.
    working_series.loc[
        day1_date
    ] = day1_prediction

    # --------------------------------------------------------
    # Day +2 ... Day +8 = Bridge model
    # --------------------------------------------------------

    for horizon_day in range(
        2,
        FORECAST_HORIZON + 1,
    ):
        forecast_date = (
            anchor_date
            + pd.Timedelta(
                days=horizon_day
            )
        )

        feature_row = (
            create_bridge_feature_row(
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

        predictions.append(
            {
                "horizon_day":
                    horizon_day,

                "forecast_date":
                    forecast_date,

                "prediction":
                    prediction,

                "forecast_mode":
                    "bridge",

                "strategy":
                    "ridge",
            }
        )

        working_series.loc[
            forecast_date
        ] = prediction

    return predictions


# ============================================================
# Valid Backtest Origins
# ============================================================

def get_backtest_origins(
    region_demand_df,
    region_feature_df,
):
    latest_real_date = (
        region_demand_df.loc[
            region_demand_df[
                "is_real"
            ]
            == 1,
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

    candidate_origins = pd.date_range(
        start=BACKTEST_START,
        end=latest_possible_origin,
        freq=f"{BACKTEST_STEP_DAYS}D",
    )

    demand_lookup = (
        region_demand_df
        .set_index("Date")
    )

    feature_dates = set(
        region_feature_df[
            "Date"
        ]
    )

    valid_origins = []

    for origin in candidate_origins:
        if (
            origin
            not in demand_lookup.index
        ):
            continue

        if (
            demand_lookup.loc[
                origin,
                "is_real",
            ]
            != 1
        ):
            continue

        future_dates = pd.date_range(
            start=(
                origin
                + pd.Timedelta(days=1)
            ),
            periods=FORECAST_HORIZON,
            freq="D",
        )

        if not future_dates.isin(
            demand_lookup.index
        ).all():
            continue

        future_rows = (
            demand_lookup.loc[
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

        # Ridge Anchor regions require
        # a Day+1 feature row.
        region = (
            region_demand_df[
                "region"
            ]
            .iloc[0]
        )

        if (
            ANCHOR_STRATEGIES[
                region
            ]
            == "ridge"
            and origin
            not in feature_dates
        ):
            continue

        valid_origins.append(
            origin
        )

    return valid_origins


# ============================================================
# Region Backtest
# ============================================================

def backtest_region(
    region,
    region_demand_df,
    region_feature_df,
):
    logger.info(
        f"Starting Hybrid backtest: "
        f"{region}"
    )

    minimum_date = (
        region_demand_df[
            "Date"
        ].min()
    )

    maximum_date = (
        region_demand_df[
            "Date"
        ].max()
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
        region_demand_df=
            region_demand_df,
        region_feature_df=
            region_feature_df,
    )

    logger.info(
        f"{region} valid origins: "
        f"{len(origins)}"
    )

    demand_lookup = (
        region_demand_df
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
            f"{anchor_date.date()}"
        )

        # ----------------------------------------------------
        # Train Bridge model using data available
        # through the anchor date
        # ----------------------------------------------------

        bridge_training_df = (
            build_bridge_training_dataset(
                region_df=
                    region_demand_df,

                training_end_date=
                    anchor_date,

                holiday_dates=
                    holiday_dates,
            )
        )

        if bridge_training_df.empty:
            continue

        X_bridge = (
            bridge_training_df[
                BRIDGE_FEATURE_COLUMNS
            ]
        )

        y_bridge = (
            bridge_training_df[
                "target_demand"
            ]
        )

        bridge_model = (
            build_ridge_model()
        )

        bridge_model.fit(
            X_bridge,
            y_bridge,
        )

        # ----------------------------------------------------
        # Hybrid Day+1 Anchor + Day+2...8 Bridge
        # ----------------------------------------------------

        forecasts = (
            generate_hybrid_forecast(
                region=region,
                anchor_date=
                    anchor_date,
                region_demand_df=
                    region_demand_df,
                region_feature_df=
                    region_feature_df,
                bridge_model=
                    bridge_model,
                holiday_dates=
                    holiday_dates,
            )
        )

        anchor_actual = float(
            demand_lookup.loc[
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
                demand_lookup.loc[
                    forecast_date,
                    "regional_demand_mw",
                ]
            )

            hybrid_prediction = float(
                forecast[
                    "prediction"
                ]
            )

            # Simple persistence baseline
            # for comparison across the full gap.
            baseline_prediction = (
                anchor_actual
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

                    "forecast_mode":
                        forecast[
                            "forecast_mode"
                        ],

                    "strategy":
                        forecast[
                            "strategy"
                        ],

                    "actual_demand_mw":
                        actual,

                    "hybrid_prediction_mw":
                        hybrid_prediction,

                    "baseline_prediction_mw":
                        baseline_prediction,

                    "hybrid_absolute_error_mw":
                        abs(
                            hybrid_prediction
                            - actual
                        ),

                    "baseline_absolute_error_mw":
                        abs(
                            baseline_prediction
                            - actual
                        ),

                    "hybrid_percentage_error":
                        (
                            abs(
                                hybrid_prediction
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
# Horizon Summary
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

        hybrid = group[
            "hybrid_prediction_mw"
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

                "hybrid_mae":
                    mean_absolute_error(
                        actual,
                        hybrid,
                    ),

                "hybrid_rmse":
                    (
                        mean_squared_error(
                            actual,
                            hybrid,
                        )
                        ** 0.5
                    ),

                "hybrid_mape":
                    calculate_mape(
                        actual,
                        hybrid,
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
# Region Comparison
# ============================================================

def build_comparison(
    predictions,
):
    rows = []

    for region, group in (
        predictions.groupby(
            "region"
        )
    ):
        hybrid_mape = calculate_mape(
            group[
                "actual_demand_mw"
            ],
            group[
                "hybrid_prediction_mw"
            ],
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

        # Bridge-only section:
        # Day+2 onward.
        bridge_group = group[
            group[
                "horizon_day"
            ]
            >= 2
        ]

        bridge_only_mape = (
            calculate_mape(
                bridge_group[
                    "actual_demand_mw"
                ],
                bridge_group[
                    "hybrid_prediction_mw"
                ],
            )
        )

        rows.append(
            {
                "region":
                    region,

                "anchor_strategy":
                    ANCHOR_STRATEGIES[
                        region
                    ],

                "hybrid_mape":
                    hybrid_mape,

                "bridge_day2_to_day8_mape":
                    bridge_only_mape,

                "baseline_mape":
                    baseline_mape,

                "hybrid_improvement_mape_points":
                    (
                        baseline_mape
                        - hybrid_mape
                    ),

                "recommended":
                    (
                        "hybrid"
                        if hybrid_mape
                        < baseline_mape
                        else "baseline"
                    ),
            }
        )

    comparison = pd.DataFrame(
        rows
    )

    return (
        comparison
        .sort_values(
            "hybrid_improvement_mape_points",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# Overall Summary
# ============================================================

def print_overall_summary(
    predictions,
):
    actual = predictions[
        "actual_demand_mw"
    ]

    hybrid = predictions[
        "hybrid_prediction_mw"
    ]

    baseline = predictions[
        "baseline_prediction_mw"
    ]

    hybrid_mae = (
        mean_absolute_error(
            actual,
            hybrid,
        )
    )

    hybrid_rmse = (
        mean_squared_error(
            actual,
            hybrid,
        )
        ** 0.5
    )

    hybrid_mape = (
        calculate_mape(
            actual,
            hybrid,
        )
    )

    baseline_mae = (
        mean_absolute_error(
            actual,
            baseline,
        )
    )

    baseline_rmse = (
        mean_squared_error(
            actual,
            baseline,
        )
        ** 0.5
    )

    baseline_mape = (
        calculate_mape(
            actual,
            baseline,
        )
    )

    print(
        "\nOVERALL HYBRID REGIONAL SUMMARY\n"
    )

    print(
        f"Hybrid MAE:      "
        f"{hybrid_mae:.2f} MW"
    )

    print(
        f"Hybrid RMSE:     "
        f"{hybrid_rmse:.2f} MW"
    )

    print(
        f"Hybrid MAPE:     "
        f"{hybrid_mape:.2f}%"
    )

    print()

    print(
        f"Baseline MAE:    "
        f"{baseline_mae:.2f} MW"
    )

    print(
        f"Baseline RMSE:   "
        f"{baseline_rmse:.2f} MW"
    )

    print(
        f"Baseline MAPE:   "
        f"{baseline_mape:.2f}%"
    )


# ============================================================
# Main
# ============================================================

def main():
    logger.info(
        "Starting Hybrid Regional "
        "Forecast backtest"
    )

    demand_df = (
        load_processed_regional_data()
    )

    feature_df = (
        load_anchor_feature_data()
    )

    all_results = []

    regions = sorted(
        demand_df[
            "region"
        ].unique()
    )

    for region in regions:
        region_demand_df = (
            demand_df[
                demand_df[
                    "region"
                ]
                == region
            ]
            .copy()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        region_feature_df = (
            feature_df[
                feature_df[
                    "region"
                ]
                == region
            ]
            .copy()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        result = backtest_region(
            region=region,
            region_demand_df=
                region_demand_df,
            region_feature_df=
                region_feature_df,
        )

        if not result.empty:
            all_results.append(
                result
            )

    if not all_results:
        raise ValueError(
            "Hybrid regional backtest "
            "produced no predictions"
        )

    predictions = pd.concat(
        all_results,
        ignore_index=True,
    )

    summary = build_summary(
        predictions
    )

    comparison = build_comparison(
        predictions
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
        "\nHYBRID REGIONAL MODEL COMPARISON\n"
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
        "\nHYBRID REGIONAL HORIZON SUMMARY\n"
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
        "Hybrid Regional Forecast "
        "backtest completed successfully"
    )


if __name__ == "__main__":
    main()