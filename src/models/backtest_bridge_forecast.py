from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger


logger = get_logger("backtest_bridge_forecast")


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = (
    "data/processed/area_wise_demand_processed.csv"
)

PREDICTIONS_PATH = Path(
    "artifacts/bridge_backtest_predictions.csv"
)

SUMMARY_PATH = Path(
    "artifacts/bridge_backtest_summary.csv"
)


FORECAST_HORIZON = 14

RIDGE_ALPHA = 0.01

BACKTEST_START = pd.Timestamp(
    "2025-01-01"
)

# One forecast origin every 7 days
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

    # A date is considered real only when
    # all 9 regional rows came from BPDB.
    national["is_real"] = (
        national["imputed_rows"] == 0
    ).astype(int)

    logger.info(
        f"National rows: {len(national)}"
    )

    logger.info(
        "Date range: "
        f"{national['Date'].min().date()} "
        "to "
        f"{national['Date'].max().date()}"
    )

    logger.info(
        "Real observation days: "
        f"{national['is_real'].sum()}"
    )

    return national


# ============================================================
# Calendar
# ============================================================

def build_holiday_set(
    min_date,
    max_date,
):
    start_year = min_date.year
    end_year = max_date.year

    bd_holidays = (
        holidays.country_holidays(
            "BD",
            years=range(
                start_year,
                end_year + 1,
            ),
        )
    )

    return set(
        bd_holidays.keys()
    )


def add_calendar_values(
    feature_row,
    target_date,
    holiday_dates,
    minimum_date,
):
    feature_row[
        "day_of_week"
    ] = target_date.dayofweek

    feature_row[
        "month"
    ] = target_date.month

    feature_row[
        "day_of_month"
    ] = target_date.day

    feature_row[
        "day_of_year"
    ] = target_date.dayofyear

    feature_row[
        "is_weekend"
    ] = int(
        target_date.dayofweek
        in [4, 5]
    )

    feature_row[
        "is_holiday"
    ] = int(
        target_date.date()
        in holiday_dates
    )

    feature_row[
        "trend_days"
    ] = (
        target_date
        - minimum_date
    ).days

    return feature_row


# ============================================================
# Feature Creation
# ============================================================

def create_feature_row(
    demand_series,
    target_date,
    holiday_dates,
    minimum_date,
):
    """
    demand_series:
        pandas Series indexed by Date.

    target_date:
        Date whose demand we want to predict.

    Only dates before target_date are used.
    """

    required_offsets = [
        1,
        2,
        3,
        7,
        14,
        21,
        30,
    ]

    values = {}

    for lag in required_offsets:
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

    # --------------------------------------------------------
    # Rolling statistics
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
            - pd.Timedelta(days=1)
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
            demand_series
            .reindex(
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

    values = add_calendar_values(
        values,
        target_date,
        holiday_dates,
        minimum_date,
    )

    return values


# ============================================================
# Training Dataset
# ============================================================

def build_training_dataset(
    national,
    training_end_date,
    holiday_dates,
):
    logger.debug(
        f"Building bridge training data "
        f"through {training_end_date.date()}"
    )

    historical = national[
        national["Date"]
        <= training_end_date
    ].copy()

    demand_series = (
        historical
        .set_index("Date")[
            "total_demand"
        ]
        .sort_index()
    )

    minimum_date = (
        national["Date"].min()
    )

    rows = []

    for _, row in (
        historical.iterrows()
    ):
        target_date = row["Date"]

        # Target must be a real BPDB observation.
        if row["is_real"] != 1:
            continue

        feature_row = (
            create_feature_row(
                demand_series=demand_series,
                target_date=target_date,
                holiday_dates=holiday_dates,
                minimum_date=minimum_date,
            )
        )

        if feature_row is None:
            continue

        feature_row[
            "target_demand"
        ] = float(
            row["total_demand"]
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

    return training_df


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
    national,
    anchor_date,
    horizon,
    holiday_dates,
):
    """
    Forecast Day+1 ... Day+horizon.

    Real/imputed historical demand is available
    through anchor_date.

    Every generated prediction is inserted into
    the working series and may be used by the
    following forecast day.
    """

    historical = national[
        national["Date"]
        <= anchor_date
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

    predictions = []

    for horizon_day in range(
        1,
        horizon + 1,
    ):
        forecast_date = (
            anchor_date
            + pd.Timedelta(
                days=horizon_day
            )
        )

        feature_row = (
            create_feature_row(
                demand_series=working_series,
                target_date=forecast_date,
                holiday_dates=holiday_dates,
                minimum_date=minimum_date,
            )
        )

        if feature_row is None:
            raise ValueError(
                "Unable to create recursive "
                f"features for {forecast_date.date()}"
            )

        X = pd.DataFrame(
            [feature_row]
        )[FEATURE_COLUMNS]

        prediction = float(
            model.predict(X)[0]
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

        # Critical recursive step:
        # prediction becomes history for
        # subsequent forecast days.
        working_series.loc[
            forecast_date
        ] = prediction

    return predictions


# ============================================================
# Backtest Origins
# ============================================================

def get_backtest_origins(
    national,
):
    latest_real_date = (
        national.loc[
            national["is_real"] == 1,
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

    valid_origins = []

    national_lookup = (
        national
        .set_index("Date")
    )

    for origin in origins:
        if origin not in national_lookup.index:
            continue

        # Anchor itself should be real.
        if (
            national_lookup.loc[
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
            national_lookup.index
        ).all():
            continue

        future_rows = (
            national_lookup.loc[
                future_dates
            ]
        )

        # For backtesting we require real
        # actual demand for every horizon.
        if not (
            future_rows["is_real"] == 1
        ).all():
            continue

        valid_origins.append(
            origin
        )

    logger.info(
        f"Valid backtest origins: "
        f"{len(valid_origins)}"
    )

    return valid_origins


# ============================================================
# Backtest
# ============================================================

def run_backtest(
    national,
):
    holiday_dates = (
        build_holiday_set(
            national["Date"].min(),
            national["Date"].max()
            + pd.Timedelta(days=7),
        )
    )

    origins = get_backtest_origins(
        national
    )

    if not origins:
        raise ValueError(
            "No valid backtest origins found"
        )

    national_lookup = (
        national
        .set_index("Date")
    )

    results = []

    for index, anchor_date in enumerate(
        origins,
        start=1,
    ):
        logger.info(
            f"Backtest {index}/{len(origins)} | "
            f"Anchor: {anchor_date.date()}"
        )

        training_df = (
            build_training_dataset(
                national=national,
                training_end_date=anchor_date,
                holiday_dates=holiday_dates,
            )
        )

        if training_df.empty:
            logger.warning(
                "Training dataset empty for "
                f"{anchor_date.date()}"
            )

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

        recursive_predictions = (
            recursive_forecast(
                model=model,
                national=national,
                anchor_date=anchor_date,
                horizon=FORECAST_HORIZON,
                holiday_dates=holiday_dates,
            )
        )

        anchor_demand = float(
            national_lookup.loc[
                anchor_date,
                "total_demand",
            ]
        )

        for prediction_row in (
            recursive_predictions
        ):
            forecast_date = (
                prediction_row[
                    "forecast_date"
                ]
            )

            actual = float(
                national_lookup.loc[
                    forecast_date,
                    "total_demand",
                ]
            )

            bridge_prediction = float(
                prediction_row[
                    "prediction"
                ]
            )

            # Persistence baseline:
            # assume demand stays at
            # the last real anchor value.
            baseline_prediction = (
                anchor_demand
            )

            results.append(
                {
                    "anchor_date":
                        anchor_date,

                    "forecast_date":
                        forecast_date,

                    "horizon_day":
                        prediction_row[
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

    for horizon_day in sorted(
        predictions[
            "horizon_day"
        ].unique()
    ):
        horizon_df = predictions[
            predictions[
                "horizon_day"
            ]
            == horizon_day
        ]

        actual = horizon_df[
            "actual_demand_mw"
        ]

        bridge = horizon_df[
            "bridge_prediction_mw"
        ]

        baseline = horizon_df[
            "baseline_prediction_mw"
        ]

        bridge_mae = (
            mean_absolute_error(
                actual,
                bridge,
            )
        )

        bridge_rmse = (
            mean_squared_error(
                actual,
                bridge,
            )
            ** 0.5
        )

        bridge_mape = (
            calculate_mape(
                actual,
                bridge,
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

        rows.append(
            {
                "horizon_day":
                    horizon_day,

                "samples":
                    len(
                        horizon_df
                    ),

                "bridge_mae":
                    bridge_mae,

                "bridge_rmse":
                    bridge_rmse,

                "bridge_mape":
                    bridge_mape,

                "baseline_mae":
                    baseline_mae,

                "baseline_rmse":
                    baseline_rmse,

                "baseline_mape":
                    baseline_mape,

                "bridge_mape_improvement_points":
                    (
                        baseline_mape
                        - bridge_mape
                    ),
            }
        )

    return pd.DataFrame(
        rows
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

    bridge = predictions[
        "bridge_prediction_mw"
    ]

    baseline = predictions[
        "baseline_prediction_mw"
    ]

    bridge_mae = mean_absolute_error(
        actual,
        bridge,
    )

    bridge_rmse = (
        mean_squared_error(
            actual,
            bridge,
        )
        ** 0.5
    )

    bridge_mape = calculate_mape(
        actual,
        bridge,
    )

    baseline_mae = mean_absolute_error(
        actual,
        baseline,
    )

    baseline_rmse = (
        mean_squared_error(
            actual,
            baseline,
        )
        ** 0.5
    )

    baseline_mape = calculate_mape(
        actual,
        baseline,
    )

    print(
        "\nOVERALL BRIDGE BACKTEST SUMMARY\n"
    )

    print(
        f"Bridge MAE:      "
        f"{bridge_mae:.2f} MW"
    )

    print(
        f"Bridge RMSE:     "
        f"{bridge_rmse:.2f} MW"
    )

    print(
        f"Bridge MAPE:     "
        f"{bridge_mape:.2f}%"
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
        "Starting Bridge Forecast backtest"
    )

    national = load_national_data()

    predictions = run_backtest(
        national
    )

    if predictions.empty:
        raise ValueError(
            "Bridge backtest produced "
            "no predictions"
        )

    summary = build_summary(
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

    print(
        "\nBRIDGE FORECAST HORIZON SUMMARY\n"
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
        "Bridge backtest predictions "
        f"saved to: {PREDICTIONS_PATH}"
    )

    logger.info(
        "Bridge backtest summary "
        f"saved to: {SUMMARY_PATH}"
    )

    logger.info(
        "Bridge Forecast backtest "
        "completed successfully"
    )


if __name__ == "__main__":
    main()