import pandas as pd

from src.logger import get_logger


logger = get_logger("build_features")

INPUT_PATH = "data/processed/national_daily_demand.csv"
OUTPUT_PATH = "data/processed/model_features.csv"


def add_exact_lag_feature(df, days, source_col, new_col):
    logger.info(
        f"Creating exact {days}-day lag feature: {new_col}"
    )

    lag_df = df[["Date", source_col]].copy()

    lag_df["Date"] = (
        lag_df["Date"] + pd.Timedelta(days=days)
    )

    lag_df = lag_df.rename(
        columns={
            source_col: new_col
        }
    )

    return df.merge(
        lag_df,
        on="Date",
        how="left"
    )


def build_features():
    logger.info("Loading national daily demand dataset")

    df = pd.read_csv(INPUT_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    logger.info(f"Initial rows: {len(df)}")

    # --------------------------------
    # Calendar Features
    # --------------------------------

    logger.info("Creating calendar features")

    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["day_of_month"] = df["Date"].dt.day

    df["is_weekend"] = (
        df["day_of_week"]
        .isin([5, 6])
        .astype(int)
    )

    # --------------------------------
    # Exact Calendar-Day Lag Features
    # --------------------------------

    df = add_exact_lag_feature(
        df,
        days=1,
        source_col="total_demand",
        new_col="lag_1_day"
    )

    df = add_exact_lag_feature(
        df,
        days=7,
        source_col="total_demand",
        new_col="lag_7_day"
    )

    df = add_exact_lag_feature(
        df,
        days=14,
        source_col="total_demand",
        new_col="lag_14_day"
    )

    df = add_exact_lag_feature(
        df,
        days=1,
        source_col="total_load_shed",
        new_col="load_shed_lag_1_day"
    )

    # --------------------------------
    # Rolling Features
    # --------------------------------

    logger.info(
        "Creating exact calendar-based rolling demand features"
    )

    demand_series = (
        df[["Date", "total_demand"]]
        .set_index("Date")
        .asfreq("D")
    )

    demand_series["rolling_7_day_mean"] = (
        demand_series["total_demand"]
        .shift(1)
        .rolling(
            window=7,
            min_periods=7
        )
        .mean()
    )

    demand_series["rolling_14_day_mean"] = (
        demand_series["total_demand"]
        .shift(1)
        .rolling(
            window=14,
            min_periods=14
        )
        .mean()
    )

    demand_series["rolling_30_day_mean"] = (
        demand_series["total_demand"]
        .shift(1)
        .rolling(
            window=30,
            min_periods=30
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
        how="left"
    )

    # --------------------------------
    # Remove Invalid Feature Rows
    # --------------------------------

    feature_columns = [
        "lag_1_day",
        "lag_7_day",
        "lag_14_day",
        "rolling_7_day_mean",
        "rolling_14_day_mean",
        "rolling_30_day_mean",
        "load_shed_lag_1_day",
    ]

    logger.info(
        "Removing rows with unavailable historical features"
    )

    before = len(df)

    df = df.dropna(
        subset=feature_columns
    ).copy()

    removed = before - len(df)

    logger.info(
        f"Rows removed because of missing lag/rolling history: {removed}"
    )

    logger.info(
        f"Final feature rows: {len(df)}"
    )

    # --------------------------------
    # Validation
    # --------------------------------

    logger.info("Validating feature dataset")

    if df.empty:
        raise ValueError(
            "Feature dataset is empty"
        )

    if df[feature_columns].isna().any().any():
        raise ValueError(
            "Missing values remain in model features"
        )

    if df["next_day_total_demand"].isna().any():
        raise ValueError(
            "Target contains missing values"
        )

    duplicate_count = df.duplicated(
        subset=["Date"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Duplicate dates found: {duplicate_count}"
        )

    # --------------------------------
    # Save
    # --------------------------------

    logger.info("Saving feature dataset")

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    logger.info(
        f"Feature dataset saved to: {OUTPUT_PATH}"
    )

    logger.info(
        f"Date range: "
        f"{df['Date'].min().date()} "
        f"to {df['Date'].max().date()}"
    )

    logger.info(
        "Feature engineering completed successfully"
    )


if __name__ == "__main__":
    build_features()