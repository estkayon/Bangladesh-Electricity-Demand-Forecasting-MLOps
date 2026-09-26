import pandas as pd
import holidays

from src.logger import get_logger


logger = get_logger("build_inference_features")


PROCESSED_DATA_PATH = (
    "data/processed/area_wise_demand_processed.csv"
)

WEATHER_PATH = (
    "data/raw/dhaka_weather.csv"
)

NATIONAL_OUTPUT_PATH = (
    "data/processed/national_inference_features.csv"
)

REGIONAL_OUTPUT_PATH = (
    "data/processed/regional_inference_features.csv"
)


# ============================================================
# Weather
# ============================================================

def load_weather():
    logger.info(
        "Loading weather dataset"
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
        f"Weather rows: {len(weather)}"
    )

    return weather


# ============================================================
# Calendar Features
# ============================================================

def add_calendar_features(df):
    logger.info(
        "Creating forecast-date calendar features"
    )

    df["forecast_date"] = (
        df["Date"]
        + pd.Timedelta(days=1)
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
    # Friday + Saturday
    df["is_weekend"] = (
        df["day_of_week"]
        .isin([4, 5])
        .astype(int)
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

    bd_holidays = (
        holidays.country_holidays(
            "BD",
            years=range(
                start_year,
                end_year + 1,
            ),
        )
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

    minimum_forecast_date = (
        df["forecast_date"]
        .min()
    )

    df["trend_days"] = (
        df["forecast_date"]
        - minimum_forecast_date
    ).dt.days

    return df


# ============================================================
# National Features
# ============================================================

def add_national_exact_lag(
    df,
    days,
    source_col,
    new_col,
):
    lag_df = df[
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


def build_national_features(
    processed,
    weather,
):
    logger.info(
        "Building national inference features"
    )

    national = (
        processed
        .groupby(
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
        f"National source rows: "
        f"{len(national)}"
    )

    national = add_calendar_features(
        national
    )

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    weather_columns = [
        "Date",
        "temperature_max_c",
        "temperature_min_c",
        "temperature_mean_c",
        "precipitation_mm",
        "rain_mm",
    ]

    national = national.merge(
        weather[
            weather_columns
        ],
        on="Date",
        how="left",
    )

    # --------------------------------------------------------
    # Exact lags
    # --------------------------------------------------------

    national = add_national_exact_lag(
        national,
        1,
        "total_demand",
        "lag_1_day",
    )

    national = add_national_exact_lag(
        national,
        7,
        "total_demand",
        "lag_7_day",
    )

    national = add_national_exact_lag(
        national,
        14,
        "total_demand",
        "lag_14_day",
    )

    national = add_national_exact_lag(
        national,
        1,
        "total_load_shed",
        "load_shed_lag_1_day",
    )

    # --------------------------------------------------------
    # Rolling features
    # --------------------------------------------------------

    national = (
        national
        .sort_values("Date")
        .set_index("Date")
        .asfreq("D")
        .reset_index()
    )

    national[
        "rolling_7_day_mean"
    ] = (
        national[
            "total_demand"
        ]
        .shift(1)
        .rolling(
            window=7,
            min_periods=7,
        )
        .mean()
    )

    national[
        "rolling_14_day_mean"
    ] = (
        national[
            "total_demand"
        ]
        .shift(1)
        .rolling(
            window=14,
            min_periods=14,
        )
        .mean()
    )

    national[
        "rolling_30_day_mean"
    ] = (
        national[
            "total_demand"
        ]
        .shift(1)
        .rolling(
            window=30,
            min_periods=30,
        )
        .mean()
    )

    # --------------------------------------------------------
    # Model-required features
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Current observation must be real BPDB data
    # --------------------------------------------------------

    before = len(national)

    national = national[
        national["imputed_rows"] == 0
    ].copy()

    national = national.dropna(
        subset=feature_columns
    ).copy()

    removed = (
        before
        - len(national)
    )

    logger.info(
        "National rows removed after "
        f"inference filtering: {removed}"
    )

    national = (
        national
        .sort_values("Date")
        .reset_index(drop=True)
    )

    national.to_csv(
        NATIONAL_OUTPUT_PATH,
        index=False,
    )

    logger.info(
        f"National inference rows: "
        f"{len(national)}"
    )

    if not national.empty:
        latest = national.iloc[-1]

        logger.info(
            "Latest national observation: "
            f"{latest['Date'].date()}"
        )

        logger.info(
            "Latest national forecast date: "
            f"{latest['forecast_date'].date()}"
        )

    logger.info(
        "National inference features saved to: "
        f"{NATIONAL_OUTPUT_PATH}"
    )

    return national


# ============================================================
# Regional Features
# ============================================================

def add_regional_exact_lag(
    df,
    days,
    source_col,
    new_col,
):
    lag_df = df[
        [
            "Date",
            "region",
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
        on=[
            "Date",
            "region",
        ],
        how="left",
    )


def create_regional_rolling(
    df,
):
    logger.info(
        "Creating regional inference rolling features"
    )

    outputs = []

    regions = sorted(
        df["region"].unique()
    )

    for region in regions:
        region_df = (
            df[
                df["region"] == region
            ][
                [
                    "Date",
                    "regional_demand_mw",
                ]
            ]
            .copy()
            .sort_values("Date")
            .set_index("Date")
            .asfreq("D")
        )

        region_df[
            "rolling_7_day_mean"
        ] = (
            region_df[
                "regional_demand_mw"
            ]
            .shift(1)
            .rolling(
                window=7,
                min_periods=7,
            )
            .mean()
        )

        region_df[
            "rolling_14_day_mean"
        ] = (
            region_df[
                "regional_demand_mw"
            ]
            .shift(1)
            .rolling(
                window=14,
                min_periods=14,
            )
            .mean()
        )

        region_df[
            "rolling_30_day_mean"
        ] = (
            region_df[
                "regional_demand_mw"
            ]
            .shift(1)
            .rolling(
                window=30,
                min_periods=30,
            )
            .mean()
        )

        region_df = (
            region_df[
                [
                    "rolling_7_day_mean",
                    "rolling_14_day_mean",
                    "rolling_30_day_mean",
                ]
            ]
            .reset_index()
        )

        region_df["region"] = region

        outputs.append(
            region_df
        )

    return pd.concat(
        outputs,
        ignore_index=True,
    )


def build_regional_features(
    processed,
    weather,
):
    logger.info(
        "Building regional inference features"
    )

    regional = processed.rename(
        columns={
            "Zone Name":
                "region",
            "Demand (MW)":
                "regional_demand_mw",
            "Load shed (MW)":
                "regional_load_shed_mw",
        }
    ).copy()

    regional = (
        regional
        .sort_values(
            [
                "region",
                "Date",
            ]
        )
        .reset_index(drop=True)
    )

    regional = add_calendar_features(
        regional
    )

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    weather_columns = [
        "Date",
        "temperature_max_c",
        "temperature_min_c",
        "temperature_mean_c",
        "precipitation_mm",
        "rain_mm",
    ]

    regional = regional.merge(
        weather[
            weather_columns
        ],
        on="Date",
        how="left",
    )

    # --------------------------------------------------------
    # Exact lags
    # --------------------------------------------------------

    regional = add_regional_exact_lag(
        regional,
        1,
        "regional_demand_mw",
        "lag_1_day",
    )

    regional = add_regional_exact_lag(
        regional,
        7,
        "regional_demand_mw",
        "lag_7_day",
    )

    regional = add_regional_exact_lag(
        regional,
        14,
        "regional_demand_mw",
        "lag_14_day",
    )

    regional = add_regional_exact_lag(
        regional,
        1,
        "regional_load_shed_mw",
        "load_shed_lag_1_day",
    )

    # --------------------------------------------------------
    # Rolling features
    # --------------------------------------------------------

    rolling_df = (
        create_regional_rolling(
            regional
        )
    )

    regional = regional.merge(
        rolling_df,
        on=[
            "Date",
            "region",
        ],
        how="left",
    )

    feature_columns = [
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

    # --------------------------------------------------------
    # Current observation must be real
    # --------------------------------------------------------

    before = len(regional)

    regional = regional[
        regional["is_imputed"] == 0
    ].copy()

    regional = regional.dropna(
        subset=feature_columns
    ).copy()

    removed = (
        before
        - len(regional)
    )

    logger.info(
        "Regional rows removed after "
        f"inference filtering: {removed}"
    )

    regional = (
        regional
        .sort_values(
            [
                "region",
                "Date",
            ]
        )
        .reset_index(drop=True)
    )

    regional.to_csv(
        REGIONAL_OUTPUT_PATH,
        index=False,
    )

    logger.info(
        f"Regional inference rows: "
        f"{len(regional)}"
    )

    logger.info(
        "Latest observation per region:"
    )

    latest_rows = (
        regional
        .sort_values("Date")
        .groupby(
            "region",
            as_index=False,
        )
        .tail(1)
        .sort_values("region")
    )

    for _, row in (
        latest_rows.iterrows()
    ):
        logger.info(
            f"{row['region']}: "
            f"observation="
            f"{row['Date'].date()}, "
            f"forecast="
            f"{row['forecast_date'].date()}"
        )

    logger.info(
        "Regional inference features saved to: "
        f"{REGIONAL_OUTPUT_PATH}"
    )

    return regional


# ============================================================
# Main
# ============================================================

def main():
    logger.info(
        "Starting live inference feature pipeline"
    )

    processed = pd.read_csv(
        PROCESSED_DATA_PATH
    )

    processed["Date"] = pd.to_datetime(
        processed["Date"]
    )

    processed = (
        processed
        .sort_values(
            [
                "Zone Name",
                "Date",
            ]
        )
        .reset_index(drop=True)
    )

    logger.info(
        f"Processed source rows: "
        f"{len(processed)}"
    )

    logger.info(
        "Processed source date range: "
        f"{processed['Date'].min().date()} "
        "to "
        f"{processed['Date'].max().date()}"
    )

    weather = load_weather()

    national = build_national_features(
        processed,
        weather,
    )

    regional = build_regional_features(
        processed,
        weather,
    )

    print(
        "\nLIVE INFERENCE FEATURE STATUS\n"
    )

    if not national.empty:
        latest_national = (
            national.iloc[-1]
        )

        print(
            "National:"
        )

        print(
            "  Latest observation date:",
            latest_national[
                "Date"
            ].date(),
        )

        print(
            "  Forecast date:",
            latest_national[
                "forecast_date"
            ].date(),
        )

    print(
        "\nRegional:"
    )

    latest_regional = (
        regional
        .sort_values("Date")
        .groupby(
            "region",
            as_index=False,
        )
        .tail(1)
        .sort_values("region")
    )

    for _, row in (
        latest_regional.iterrows()
    ):
        print(
            f"  {row['region']}: "
            f"{row['Date'].date()} "
            f"-> "
            f"{row['forecast_date'].date()}"
        )

    logger.info(
        "Live inference feature pipeline "
        "completed successfully"
    )


if __name__ == "__main__":
    main()