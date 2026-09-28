import pandas as pd
import holidays

from src.logger import get_logger


logger = get_logger("build_features")

INPUT_PATH = "data/processed/national_daily_demand.csv"
CONTEXT_PATH = "data/processed/area_wise_demand_processed.csv"
WEATHER_PATH = "data/raw/dhaka_weather.csv"
OUTPUT_PATH = "data/processed/model_features.csv"


def load_context_data():
    """
    Load the complete processed daily demand series.

    Real observations and imputed historical dates are both kept here
    because this dataset is used only for lag/rolling feature context.

    Imputed dates are NOT used as historical evaluation targets.
    """

    logger.info(
        "Loading complete processed demand context"
    )

    context = pd.read_csv(
        CONTEXT_PATH
    )

    context["Date"] = pd.to_datetime(
        context["Date"]
    )

    context = (
        context.groupby(
            "Date",
            as_index=False,
        )
        .agg(
            total_demand=(
                "Demand (MW)",
                "sum",
            ),
            total_load_shed=(
                "Load shed (MW)",
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

    logger.info(
        f"Context rows loaded: {len(context)}"
    )

    logger.info(
        "Context date range: "
        f"{context['Date'].min().date()} "
        "to "
        f"{context['Date'].max().date()}"
    )

    logger.info(
        "Fully imputed context days: "
        f"{(context['imputed_rows'] == 9).sum()}"
    )

    return context


def add_exact_lag_feature(
    df,
    context,
    days,
    source_col,
    new_col,
):
    """
    Create exact calendar-day lag feature using the complete
    processed context dataset.
    """

    logger.info(
        f"Creating exact {days}-day lag feature: "
        f"{new_col}"
    )

    lag_df = context[
        [
            "Date",
            source_col,
        ]
    ].copy()

    lag_df["Date"] = (
        lag_df["Date"]
        + pd.Timedelta(days=days)
    )

    lag_df = lag_df.rename(
        columns={
            source_col: new_col
        }
    )

    return df.merge(
        lag_df,
        on="Date",
        how="left",
    )


def load_weather_data():
    logger.info(
        "Loading Dhaka weather dataset"
    )

    weather = pd.read_csv(
        WEATHER_PATH
    )

    weather["Date"] = pd.to_datetime(
        weather["Date"]
    )

    weather = (
        weather
        .sort_values("Date")
        .drop_duplicates(
            subset=["Date"]
        )
        .reset_index(drop=True)
    )

    logger.info(
        f"Weather rows loaded: "
        f"{len(weather)}"
    )

    logger.info(
        "Weather date range: "
        f"{weather['Date'].min().date()} "
        "to "
        f"{weather['Date'].max().date()}"
    )

    return weather


def build_features():
    logger.info(
        "Loading national daily demand dataset"
    )

    df = pd.read_csv(
        INPUT_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    logger.info(
        f"Initial target rows: {len(df)}"
    )

    logger.info(
        "Target date range: "
        f"{df['Date'].min().date()} "
        "to "
        f"{df['Date'].max().date()}"
    )

    # --------------------------------------------------
    # Complete historical context
    # --------------------------------------------------

    context = load_context_data()

    # --------------------------------------------------
    # Forecast Date
    # --------------------------------------------------

    logger.info(
        "Creating next-day forecast date"
    )

    df["forecast_date"] = (
        df["Date"]
        + pd.Timedelta(days=1)
    )

    # --------------------------------------------------
    # Target-Day Calendar Features
    # --------------------------------------------------

    logger.info(
        "Creating next-day calendar features"
    )

    df["day_of_week"] = (
        df["forecast_date"]
        .dt.dayofweek
    )

    df["month"] = (
        df["forecast_date"]
        .dt.month
    )

    df["day_of_month"] = (
        df["forecast_date"]
        .dt.day
    )

    df["day_of_year"] = (
        df["forecast_date"]
        .dt.dayofyear
    )

    # Bangladesh weekend:
    # Friday = 4
    # Saturday = 5

    df["is_weekend"] = (
        df["day_of_week"]
        .isin([4, 5])
        .astype(int)
    )

    # --------------------------------------------------
    # Bangladesh Holiday Feature
    # --------------------------------------------------

    logger.info(
        "Creating Bangladesh public holiday feature"
    )

    start_year = (
        df["forecast_date"]
        .dt.year
        .min()
    )

    end_year = (
        df["forecast_date"]
        .dt.year
        .max()
    )

    bd_holidays = holidays.country_holidays(
        "BD",
        years=range(
            start_year,
            end_year + 1,
        ),
    )

    holiday_dates = set(
        bd_holidays.keys()
    )

    df["is_holiday"] = (
        df["forecast_date"]
        .dt.date
        .isin(holiday_dates)
        .astype(int)
    )

    logger.info(
        "Holiday rows identified: "
        f"{df['is_holiday'].sum()}"
    )

    # --------------------------------------------------
    # Time Trend
    # --------------------------------------------------

    logger.info(
        "Creating time trend feature"
    )

    minimum_date = (
        df["forecast_date"]
        .min()
    )

    df["trend_days"] = (
        df["forecast_date"]
        - minimum_date
    ).dt.days

    # --------------------------------------------------
    # Weather Features
    #
    # Current-day observed weather is used.
    # Future target-day actual weather is not used.
    # --------------------------------------------------

    logger.info(
        "Merging current-day Dhaka weather features"
    )

    weather = load_weather_data()

    weather_columns = [
        "Date",
        "temperature_max_c",
        "temperature_min_c",
        "temperature_mean_c",
        "precipitation_mm",
        "rain_mm",
    ]

    weather = weather[
        weather_columns
    ].copy()

    df = df.merge(
        weather,
        on="Date",
        how="left",
    )

    weather_missing = (
        df[
            [
                "temperature_max_c",
                "temperature_min_c",
                "temperature_mean_c",
                "precipitation_mm",
                "rain_mm",
            ]
        ]
        .isna()
        .sum()
        .sum()
    )

    logger.info(
        "Missing weather values after merge: "
        f"{weather_missing}"
    )

    # --------------------------------------------------
    # Exact Demand Lag Features
    #
    # IMPORTANT:
    # Lags are generated from complete processed context,
    # not from the sparse real-target dataset.
    # --------------------------------------------------

    df = add_exact_lag_feature(
        df=df,
        context=context,
        days=1,
        source_col="total_demand",
        new_col="lag_1_day",
    )

    df = add_exact_lag_feature(
        df=df,
        context=context,
        days=7,
        source_col="total_demand",
        new_col="lag_7_day",
    )

    df = add_exact_lag_feature(
        df=df,
        context=context,
        days=14,
        source_col="total_demand",
        new_col="lag_14_day",
    )

    df = add_exact_lag_feature(
        df=df,
        context=context,
        days=1,
        source_col="total_load_shed",
        new_col="load_shed_lag_1_day",
    )

    # --------------------------------------------------
    # Rolling Demand Features
    #
    # Rolling windows are generated from complete
    # processed calendar-day context.
    # --------------------------------------------------

    logger.info(
        "Creating calendar-based rolling features "
        "from complete processed context"
    )

    demand_series = (
        context[
            [
                "Date",
                "total_demand",
            ]
        ]
        .set_index("Date")
        .sort_index()
        .asfreq("D")
    )

    demand_series[
        "rolling_7_day_mean"
    ] = (
        demand_series[
            "total_demand"
        ]
        .shift(1)
        .rolling(
            window=7,
            min_periods=7,
        )
        .mean()
    )

    demand_series[
        "rolling_14_day_mean"
    ] = (
        demand_series[
            "total_demand"
        ]
        .shift(1)
        .rolling(
            window=14,
            min_periods=14,
        )
        .mean()
    )

    demand_series[
        "rolling_30_day_mean"
    ] = (
        demand_series[
            "total_demand"
        ]
        .shift(1)
        .rolling(
            window=30,
            min_periods=30,
        )
        .mean()
    )

    rolling_features = (
        demand_series[
            [
                "rolling_7_day_mean",
                "rolling_14_day_mean",
                "rolling_30_day_mean",
            ]
        ]
        .reset_index()
    )

    df = df.merge(
        rolling_features,
        on="Date",
        how="left",
    )

    # --------------------------------------------------
    # Feature Validation
    # --------------------------------------------------

    feature_columns = [
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

    logger.info(
        "Removing rows with unavailable features"
    )

    before = len(df)

    df = df.dropna(
        subset=feature_columns
    ).copy()

    removed = (
        before - len(df)
    )

    logger.info(
        "Rows removed because of unavailable "
        f"features: {removed}"
    )

    logger.info(
        f"Final feature rows: {len(df)}"
    )

    # --------------------------------------------------
    # Final Validation
    # --------------------------------------------------

    logger.info(
        "Validating feature dataset"
    )

    if df.empty:
        raise ValueError(
            "Feature dataset is empty"
        )

    if (
        df[
            feature_columns
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Missing values remain in model features"
        )

    if (
        df[
            "next_day_total_demand"
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            "Target contains missing values"
        )

    duplicate_count = (
        df.duplicated(
            subset=["Date"]
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Duplicate dates found: "
            f"{duplicate_count}"
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    logger.info(
        "Saving feature dataset"
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    logger.info(
        "Feature dataset saved to: "
        f"{OUTPUT_PATH}"
    )

    logger.info(
        "Date range: "
        f"{df['Date'].min().date()} "
        "to "
        f"{df['Date'].max().date()}"
    )

    logger.info(
        "Forecast date range: "
        f"{df['forecast_date'].min().date()} "
        "to "
        f"{df['forecast_date'].max().date()}"
    )

    logger.info(
        "Feature engineering completed successfully"
    )


if __name__ == "__main__":
    build_features()