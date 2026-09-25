import pandas as pd

from src.logger import get_logger


logger = get_logger("build_target")

INPUT_PATH = "data/processed/area_wise_demand_processed.csv"
OUTPUT_PATH = "data/processed/national_daily_demand.csv"

CUTOFF_DATE = "2026-09-19"


def build_target():
    logger.info("Loading processed area-wise dataset")

    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    logger.info("Aggregating zone-wise data into national daily demand")

    daily = (
        df.groupby("Date", as_index=False)
        .agg(
            total_demand=("Demand (MW)", "sum"),
            total_load_shed=("Load shed (MW)", "sum"),
            imputed_rows=("is_imputed", "sum"),
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    logger.info(f"Applying cutoff date: {CUTOFF_DATE}")

    daily = daily[
        daily["Date"] <= pd.to_datetime(CUTOFF_DATE)
    ].copy()

    fully_imputed = daily[daily["imputed_rows"] == 9]

    logger.info(
        f"Fully imputed days found: {len(fully_imputed)}"
    )

    # Create exact next-calendar-day target table
    target = daily[
        ["Date", "total_demand", "imputed_rows"]
    ].copy()

    target["Date"] = target["Date"] - pd.Timedelta(days=1)

    target = target.rename(
        columns={
            "total_demand": "next_day_total_demand",
            "imputed_rows": "next_day_imputed_rows",
        }
    )

    logger.info("Merging exact next-day demand target")

    daily = daily.merge(
        target,
        on="Date",
        how="left"
    )

    # Current day itself must contain real observations
    logger.info("Removing fully imputed feature days")

    daily = daily[
        daily["imputed_rows"] < 9
    ].copy()

    # Target day must also contain real observations
    logger.info("Removing rows whose next-day target is fully imputed")

    daily = daily[
        daily["next_day_imputed_rows"] < 9
    ].copy()

    # Remove dates without next-day target
    daily = daily.dropna(
        subset=["next_day_total_demand"]
    ).copy()

    logger.info("Validating target dataset")

    if daily.empty:
        raise ValueError("Target dataset is empty")

    if daily["next_day_total_demand"].isna().any():
        raise ValueError(
            "Missing values found in next_day_total_demand"
        )

    # Verify target is genuinely next calendar day
    daily = daily.sort_values("Date").reset_index(drop=True)

    logger.info(
        f"Final rows: {len(daily)}"
    )

    logger.info(
        f"Date range: "
        f"{daily['Date'].min().date()} "
        f"to {daily['Date'].max().date()}"
    )

    logger.info(
        f"Maximum current-day imputed rows: "
        f"{daily['imputed_rows'].max()}"
    )

    logger.info(
        f"Maximum target-day imputed rows: "
        f"{daily['next_day_imputed_rows'].max()}"
    )

    logger.info("Saving target dataset")

    daily.to_csv(
        OUTPUT_PATH,
        index=False
    )

    logger.info(
        f"Dataset saved to: {OUTPUT_PATH}"
    )

    logger.info(
        "Target dataset creation completed successfully"
    )


if __name__ == "__main__":
    build_target()