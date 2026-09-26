import pandas as pd
import holidays

from src.logger import get_logger


logger = get_logger("build_regional_features")

INPUT_PATH = "data/processed/area_wise_demand_processed.csv"
WEATHER_PATH = "data/raw/dhaka_weather.csv"
OUTPUT_PATH = "data/processed/regional_model_features.csv"


def load_weather_data():
    logger.info("Loading Dhaka weather dataset")

    weather = pd.read_csv(WEATHER_PATH)
    weather["Date"] = pd.to_datetime(weather["Date"])

    weather = (
        weather
        .sort_values("Date")
        .drop_duplicates(subset=["Date"])
        .reset_index(drop=True)
    )

    logger.info(f"Weather rows: {len(weather)}")

    return weather


def add_exact_regional_lag(
    df,
    days,
    source_col,
    new_col,
):
    logger.info(
        f"Creating exact {days}-day regional lag: {new_col}"
    )

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
        on=["Date", "region"],
        how="left",
    )


def create_regional_rolling_features(df):
    logger.info(
        "Creating regional rolling demand features"
    )

    outputs = []

    for region in sorted(df["region"].unique()):
        logger.info(
            f"Processing rolling features for: {region}"
        )

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

        region_df["rolling_7_day_mean"] = (
            region_df["regional_demand_mw"]
            .shift(1)
            .rolling(
                window=7,
                min_periods=7,
            )
            .mean()
        )

        region_df["rolling_14_day_mean"] = (
            region_df["regional_demand_mw"]
            .shift(1)
            .rolling(
                window=14,
                min_periods=14,
            )
            .mean()
        )

        region_df["rolling_30_day_mean"] = (
            region_df["regional_demand_mw"]
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

        outputs.append(region_df)

    return pd.concat(
        outputs,
        ignore_index=True,
    )


def build_regional_features():
    logger.info(
        "Loading complete processed regional dataset"
    )

    df = pd.read_csv(INPUT_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.rename(
        columns={
            "Zone Name": "region",
            "Demand (MW)": "regional_demand_mw",
            "Load shed (MW)": "regional_load_shed_mw",
        }
    )

    df = (
        df
        .sort_values(["region", "Date"])
        .reset_index(drop=True)
    )

    logger.info(f"Initial rows: {len(df)}")
    logger.info(
        f"Regions: {df['region'].nunique()}"
    )

    # --------------------------------------------------
    # Important:
    # Features are created BEFORE removing imputed days.
    # Imputed historical values may be used as context,
    # but prediction/target dates must be real BPDB data.
    # --------------------------------------------------

    # Forecast Date
    df["forecast_date"] = (
        df["Date"]
        + pd.Timedelta(days=1)
    )

    # --------------------------------------------------
    # Target-day calendar features
    # --------------------------------------------------

    logger.info(
        "Creating target-day calendar features"
    )

    df["day_of_week"] = (
        df["forecast_date"].dt.dayofweek
    )

    df["month"] = (
        df["forecast_date"].dt.month
    )

    df["day_of_month"] = (
        df["forecast_date"].dt.day
    )

    df["day_of_year"] = (
        df["forecast_date"].dt.dayofyear
    )

    # Bangladesh weekend = Friday + Saturday
    df["is_weekend"] = (
        df["day_of_week"]
        .isin([4, 5])
        .astype(int)
    )

    # --------------------------------------------------
    # Holiday feature
    # --------------------------------------------------

    logger.info(
        "Creating Bangladesh holiday feature"
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

    # --------------------------------------------------
    # Trend
    # --------------------------------------------------

    minimum_date = (
        df["forecast_date"].min()
    )

    df["trend_days"] = (
        df["forecast_date"]
        - minimum_date
    ).dt.days

    # --------------------------------------------------
    # Weather
    # --------------------------------------------------

    logger.info(
        "Merging current-day weather features"
    )

    weather = load_weather_data()

    weather = weather[
        [
            "Date",
            "temperature_max_c",
            "temperature_min_c",
            "temperature_mean_c",
            "precipitation_mm",
            "rain_mm",
        ]
    ].copy()

    df = df.merge(
        weather,
        on="Date",
        how="left",
    )

    logger.info(
        "Missing weather values after merge: "
        f"{df[['temperature_max_c', 'temperature_min_c', 'temperature_mean_c', 'precipitation_mm', 'rain_mm']].isna().sum().sum()}"
    )

    # --------------------------------------------------
    # Historical lag features
    # --------------------------------------------------

    df = add_exact_regional_lag(
        df,
        1,
        "regional_demand_mw",
        "lag_1_day",
    )

    df = add_exact_regional_lag(
        df,
        7,
        "regional_demand_mw",
        "lag_7_day",
    )

    df = add_exact_regional_lag(
        df,
        14,
        "regional_demand_mw",
        "lag_14_day",
    )

    df = add_exact_regional_lag(
        df,
        1,
        "regional_load_shed_mw",
        "load_shed_lag_1_day",
    )

    # --------------------------------------------------
    # Rolling features
    # --------------------------------------------------

    rolling_df = (
        create_regional_rolling_features(df)
    )

    df = df.merge(
        rolling_df,
        on=["Date", "region"],
        how="left",
    )

    # --------------------------------------------------
    # Exact next-day target
    # --------------------------------------------------

    logger.info(
        "Creating exact next-day regional target"
    )

    target_df = df[
        [
            "Date",
            "region",
            "regional_demand_mw",
            "is_imputed",
        ]
    ].copy()

    target_df["Date"] = (
        target_df["Date"]
        - pd.Timedelta(days=1)
    )

    target_df = target_df.rename(
        columns={
            "regional_demand_mw":
                "next_day_demand_mw",
            "is_imputed":
                "next_day_is_imputed",
        }
    )

    df = df.merge(
        target_df,
        on=["Date", "region"],
        how="left",
    )

    # --------------------------------------------------
    # Features required by models
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Only NOW filter unreliable prediction dates
    # --------------------------------------------------

    logger.info(
        "Filtering current and target dates"
    )

    before = len(df)

    # Current observation day must be real
    df = df[
        df["is_imputed"] == 0
    ].copy()

    # Target/forecast day must be real
    df = df[
        df["next_day_is_imputed"] == 0
    ].copy()

    df = df.dropna(
        subset=[
            "next_day_demand_mw"
        ]
        + feature_columns
    ).copy()

    removed = before - len(df)

    logger.info(
        f"Rows removed after final filtering: {removed}"
    )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    duplicate_count = (
        df.duplicated(
            subset=["Date", "region"]
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Duplicate rows found: {duplicate_count}"
        )

    if df[feature_columns].isna().any().any():
        raise ValueError(
            "Missing model features remain"
        )

    if df["next_day_demand_mw"].isna().any():
        raise ValueError(
            "Missing targets remain"
        )

    df = (
        df
        .sort_values(["region", "Date"])
        .reset_index(drop=True)
    )

    logger.info(
        f"Final rows: {len(df)}"
    )

    logger.info(
        f"Regions: {df['region'].nunique()}"
    )

    logger.info(
        f"Date range: "
        f"{df['Date'].min().date()} "
        f"to "
        f"{df['Date'].max().date()}"
    )

    logger.info(
        "Feature rows per region:"
    )

    for region, count in (
        df["region"]
        .value_counts()
        .sort_index()
        .items()
    ):
        logger.info(
            f"{region}: {count}"
        )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    logger.info(
        f"Regional feature dataset saved to: "
        f"{OUTPUT_PATH}"
    )

    logger.info(
        "Regional feature engineering completed successfully"
    )


if __name__ == "__main__":
    build_regional_features()