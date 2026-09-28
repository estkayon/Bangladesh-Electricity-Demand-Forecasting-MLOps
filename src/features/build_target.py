import pandas as pd

from src.logger import get_logger


logger = get_logger("build_target")

INPUT_PATH = "data/processed/area_wise_demand_processed.csv"
OUTPUT_PATH = "data/processed/national_daily_demand.csv"


def build_target():
    logger.info(
        "Loading processed area-wise dataset"
    )

    df = pd.read_csv(INPUT_PATH)

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    logger.info(
        "Aggregating zone-wise data "
        "into national daily demand"
    )

    daily = (
        df.groupby(
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

    latest_available_date = (
        daily["Date"]
        .max()
        .normalize()
    )

    logger.info(
        "Latest available processed date: "
        f"{latest_available_date.date()}"
    )

    fully_imputed = daily[
        daily["imputed_rows"] == 9
    ]

    logger.info(
        f"Fully imputed days found: "
        f"{len(fully_imputed)}"
    )

    # --------------------------------------------------------
    # Build exact next-calendar-day target
    # --------------------------------------------------------

    target = daily[
        [
            "Date",
            "total_demand",
            "imputed_rows",
        ]
    ].copy()

    # Example:
    # demand from Sep 23 becomes the target
    # for Sep 22.
    target["Date"] = (
        target["Date"]
        - pd.Timedelta(days=1)
    )

    target = target.rename(
        columns={
            "total_demand":
                "next_day_total_demand",
            "imputed_rows":
                "next_day_imputed_rows",
        }
    )

    logger.info(
        "Merging exact next-day demand target"
    )

    daily = daily.merge(
        target,
        on="Date",
        how="left",
    )

    # --------------------------------------------------------
    # Keep only real current-day observations
    # --------------------------------------------------------

    logger.info(
        "Removing fully imputed feature days"
    )

    daily = daily[
        daily["imputed_rows"] < 9
    ].copy()

    # --------------------------------------------------------
    # Keep only real next-day targets
    # --------------------------------------------------------

    logger.info(
        "Removing rows whose next-day "
        "target is fully imputed"
    )

    daily = daily[
        (
            daily[
                "next_day_imputed_rows"
            ].notna()
        )
        &
        (
            daily[
                "next_day_imputed_rows"
            ] < 9
        )
    ].copy()

    # --------------------------------------------------------
    # Remove rows without an exact next-day target
    # --------------------------------------------------------

    daily = daily.dropna(
        subset=[
            "next_day_total_demand"
        ]
    ).copy()

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    logger.info(
        "Validating target dataset"
    )

    if daily.empty:
        raise ValueError(
            "Target dataset is empty"
        )

    if (
        daily[
            "next_day_total_demand"
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            "Missing values found in "
            "next_day_total_demand"
        )

    daily = (
        daily
        .sort_values("Date")
        .reset_index(drop=True)
    )

    logger.info(
        f"Final rows: {len(daily)}"
    )

    logger.info(
        "Date range: "
        f"{daily['Date'].min().date()} "
        "to "
        f"{daily['Date'].max().date()}"
    )

    logger.info(
        "Maximum current-day "
        "imputed rows: "
        f"{daily['imputed_rows'].max()}"
    )

    logger.info(
        "Maximum target-day "
        "imputed rows: "
        f"{daily['next_day_imputed_rows'].max()}"
    )

    logger.info(
        "Saving target dataset"
    )

    daily.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    logger.info(
        f"Dataset saved to: "
        f"{OUTPUT_PATH}"
    )

    logger.info(
        "Target dataset creation "
        "completed successfully"
    )


if __name__ == "__main__":
    build_target()