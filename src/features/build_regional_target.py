import pandas as pd

from src.logger import get_logger


logger = get_logger("build_regional_target")

INPUT_PATH = "data/processed/area_wise_demand_processed.csv"
OUTPUT_PATH = "data/processed/regional_daily_demand.csv"


def build_regional_target():
    logger.info(
        "Loading processed regional demand dataset"
    )

    df = pd.read_csv(
        INPUT_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = (
        df.sort_values(
            ["Zone Name", "Date"]
        )
        .reset_index(drop=True)
    )

    logger.info(
        f"Initial rows: {len(df)}"
    )

    logger.info(
        f"Unique regions: "
        f"{df['Zone Name'].nunique()}"
    )

    logger.info(
        "Creating exact next-calendar-day regional target"
    )

    # ---------------------------------------------
    # Build target table
    #
    # Example:
    # Original target row:
    # 2025-10-10, Dhaka, demand=...
    #
    # Move target Date back one day:
    # 2025-10-09 -> target for 2025-10-10
    #
    # Merge using:
    # Date + Zone Name
    # ---------------------------------------------

    target_df = df[
        [
            "Date",
            "Zone Name",
            "Demand (MW)",
            "is_imputed",
        ]
    ].copy()

    target_df["Date"] = (
        target_df["Date"]
        - pd.Timedelta(days=1)
    )

    target_df = target_df.rename(
        columns={
            "Demand (MW)":
                "next_day_demand_mw",
            "is_imputed":
                "next_day_is_imputed",
        }
    )

    regional = df.merge(
        target_df,
        on=[
            "Date",
            "Zone Name",
        ],
        how="left",
    )

    logger.info(
        "Filtering unreliable current-day "
        "and target-day rows"
    )

    before = len(regional)

    # Current day must be real BPDB data
    regional = regional[
        regional["is_imputed"] == 0
    ].copy()

    # Target day must also be real BPDB data
    regional = regional[
        regional[
            "next_day_is_imputed"
        ] == 0
    ].copy()

    # Remove rows without an exact next-day target
    regional = regional.dropna(
        subset=[
            "next_day_demand_mw"
        ]
    ).copy()

    removed = (
        before - len(regional)
    )

    logger.info(
        f"Rows removed: {removed}"
    )

    # ---------------------------------------------
    # Rename columns for modeling
    # ---------------------------------------------

    regional = regional.rename(
        columns={
            "Zone Name":
                "region",
            "Demand (MW)":
                "regional_demand_mw",
            "Load shed (MW)":
                "regional_load_shed_mw",
        }
    )

    # ---------------------------------------------
    # Keep clean modeling columns
    # ---------------------------------------------

    columns_to_keep = [
        "Date",
        "region",
        "regional_demand_mw",
        "regional_load_shed_mw",
        "is_imputed",
        "is_anomaly",
        "next_day_demand_mw",
        "next_day_is_imputed",
    ]

    regional = regional[
        columns_to_keep
    ].copy()

    regional = (
        regional
        .sort_values(
            ["region", "Date"]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------
    # Validation
    # ---------------------------------------------

    logger.info(
        "Validating regional target dataset"
    )

    if regional.empty:
        raise ValueError(
            "Regional target dataset is empty"
        )

    if regional[
        "next_day_demand_mw"
    ].isna().any():

        raise ValueError(
            "Missing next-day targets remain"
        )

    duplicate_count = (
        regional.duplicated(
            subset=[
                "Date",
                "region",
            ]
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Duplicate Date-region rows found: "
            f"{duplicate_count}"
        )

    if regional[
        "is_imputed"
    ].max() != 0:

        raise ValueError(
            "Imputed current-day rows remain"
        )

    if regional[
        "next_day_is_imputed"
    ].max() != 0:

        raise ValueError(
            "Imputed target-day rows remain"
        )

    logger.info(
        f"Final rows: {len(regional)}"
    )

    logger.info(
        f"Regions: "
        f"{regional['region'].nunique()}"
    )

    logger.info(
        f"Date range: "
        f"{regional['Date'].min().date()} "
        f"to "
        f"{regional['Date'].max().date()}"
    )

    logger.info(
        "Rows per region:"
    )

    for region, count in (
        regional[
            "region"
        ]
        .value_counts()
        .sort_index()
        .items()
    ):
        logger.info(
            f"{region}: {count}"
        )

    # ---------------------------------------------
    # Save
    # ---------------------------------------------

    regional.to_csv(
        OUTPUT_PATH,
        index=False
    )

    logger.info(
        f"Regional target dataset saved to: "
        f"{OUTPUT_PATH}"
    )

    logger.info(
        "Regional target generation "
        "completed successfully"
    )


if __name__ == "__main__":
    build_regional_target()