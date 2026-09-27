from pathlib import Path

import numpy as np
import pandas as pd

from src.logger import get_logger


logger = get_logger("preprocess")


# ============================================================
# Configuration
# ============================================================

RAW_DATA_PATH = Path(
    "data/raw/area_wise_demand.csv"
)

OUTPUT_PATH = Path(
    "data/processed/area_wise_demand_processed.csv"
)


REGIONS = [
    "Dhaka",
    "Chittagong",
    "Khulna",
    "Rajshahi",
    "Comilla",
    "Mymensingh",
    "Sylhet",
    "Barisal",
    "Rangpur",
]


# ============================================================
# Known Source Anomalies
# ============================================================

DEMAND_CORRECTIONS = {
    (
        pd.Timestamp("2026-08-10"),
        "Mymensingh",
    ): 1574.5,

    (
        pd.Timestamp("2023-11-02"),
        "Comilla",
    ): 1163.5,

    (
        pd.Timestamp("2024-08-01"),
        "Khulna",
    ): 1565.5,
}


LOAD_SHED_INTERPOLATION_FIXES = [
    (
        pd.Timestamp("2023-01-06"),
        "Khulna",
    ),
]


# ============================================================
# Load Raw Data
# ============================================================

def load_raw_data():
    logger.info(
        f"Loading raw BPDB data: "
        f"{RAW_DATA_PATH}"
    )

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: "
            f"{RAW_DATA_PATH}"
        )

    df = pd.read_csv(
        RAW_DATA_PATH
    )

    required_columns = {
        "Date",
        "Zone Name",
        "Demand (MW)",
        "Load shed (MW)",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df["Demand (MW)"] = pd.to_numeric(
        df["Demand (MW)"],
        errors="coerce",
    ).astype(float)

    df["Load shed (MW)"] = pd.to_numeric(
        df["Load shed (MW)"],
        errors="coerce",
    ).astype(float)

    df["is_anomaly"] = 0

    df["Zone Name"] = (
        df["Zone Name"]
        .astype(str)
        .str.strip()
    )

    df = df.dropna(
        subset=[
            "Date",
            "Zone Name",
        ]
    ).copy()

    df = df[
        df["Zone Name"].isin(
            REGIONS
        )
    ].copy()

    logger.info(
        f"Raw rows loaded: {len(df)}"
    )

    logger.info(
        "Raw date range: "
        f"{df['Date'].min().date()} "
        "to "
        f"{df['Date'].max().date()}"
    )

    return df


# ============================================================
# Remove Duplicate Region-Date Rows
# ============================================================

def remove_duplicates(df):
    duplicate_count = (
        df.duplicated(
            subset=[
                "Date",
                "Zone Name",
            ],
            keep="last",
        )
        .sum()
    )

    if duplicate_count > 0:
        logger.warning(
            f"Duplicate rows found: "
            f"{duplicate_count}"
        )

    df = (
        df
        .drop_duplicates(
            subset=[
                "Date",
                "Zone Name",
            ],
            keep="last",
        )
        .copy()
    )

    return df


# ============================================================
# Known Data Corrections
# ============================================================

def apply_known_corrections(df):
    logger.info(
        "Applying known source corrections"
    )

    for (
        correction_date,
        region,
), corrected_value in DEMAND_CORRECTIONS.items():

     mask = (
        (df["Date"] == correction_date)
        &
        (df["Zone Name"] == region)
    )

    if mask.any():
        old_value = (
            df.loc[
                mask,
                "Demand (MW)",
            ]
            .iloc[0]
        )

        df.loc[
            mask,
            "Demand (MW)",
        ] = corrected_value

        df.loc[
            mask,
            "is_anomaly",
        ] = 1

        logger.info(
            f"Demand correction: "
            f"{correction_date.date()} | "
            f"{region} | "
            f"{old_value} -> "
            f"{corrected_value}"
        )

    for (
        correction_date,
        region,
    ) in LOAD_SHED_INTERPOLATION_FIXES:
        mask = (
            (df["Date"] == correction_date)
            &
            (df["Zone Name"] == region)
        )

        if not mask.any():
            continue

        previous_date = (
            correction_date
            - pd.Timedelta(days=1)
        )

        next_date = (
            correction_date
            + pd.Timedelta(days=1)
        )

        previous_values = df.loc[
            (
                (df["Date"] == previous_date)
                &
                (df["Zone Name"] == region)
            ),
            "Load shed (MW)",
        ]

        next_values = df.loc[
            (
                (df["Date"] == next_date)
                &
                (df["Zone Name"] == region)
            ),
            "Load shed (MW)",
        ]

        if (
            not previous_values.empty
            and not next_values.empty
            and pd.notna(
                previous_values.iloc[0]
            )
            and pd.notna(
                next_values.iloc[0]
            )
        ):
            corrected_value = (
                float(
                    previous_values.iloc[0]
                )
                + float(
                    next_values.iloc[0]
                )
            ) / 2

            old_value = (
                df.loc[
                    mask,
                    "Load shed (MW)",
                ]
                .iloc[0]
            )

            df.loc[
                mask,
                "Load shed (MW)",
            ] = corrected_value

            logger.info(
                f"Load shed correction: "
                f"{correction_date.date()} | "
                f"{region} | "
                f"{old_value} -> "
                f"{corrected_value:.2f}"
            )

    return df


# ============================================================
# Build Complete Calendar Grid
# ============================================================

def build_complete_grid(df):
    start_date = (
        df["Date"].min()
        .normalize()
    )

    # IMPORTANT:
    # Dynamic end date.
    # We stop at the latest date actually
    # present in the raw BPDB dataset.
    end_date = (
        df["Date"].max()
        .normalize()
    )

    logger.info(
        "Building complete calendar grid"
    )

    logger.info(
        f"Dynamic start date: "
        f"{start_date.date()}"
    )

    logger.info(
        f"Dynamic end date: "
        f"{end_date.date()}"
    )

    all_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D",
    )

    grid = pd.MultiIndex.from_product(
        [
            all_dates,
            REGIONS,
        ],
        names=[
            "Date",
            "Zone Name",
        ],
    ).to_frame(
        index=False
    )

    logger.info(
        f"Expected complete rows: "
        f"{len(grid)}"
    )

    merged = grid.merge(
        df,
        on=[
            "Date",
            "Zone Name",
        ],
        how="left",
        indicator=True,
    )

    merged["is_imputed"] = (
        merged["_merge"]
        == "left_only"
    ).astype(int)

    merged = merged.drop(
        columns=[
            "_merge",
        ]
    )

    logger.info(
        "Missing region-date rows detected: "
        f"{merged['is_imputed'].sum()}"
    )

    return merged


# ============================================================
# Fill Missing Demand
# ============================================================

def interpolate_demand(df):
    logger.info(
        "Interpolating missing demand values"
    )

    output_frames = []

    for region in REGIONS:
        region_df = (
            df[
                df["Zone Name"] == region
            ]
            .copy()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        region_df[
            "Demand (MW)"
        ] = (
            region_df[
                "Demand (MW)"
            ]
            .interpolate(
                method="linear",
                limit_direction="both",
            )
        )

        output_frames.append(
            region_df
        )

    result = pd.concat(
        output_frames,
        ignore_index=True,
    )

    return result


# ============================================================
# Fill Missing Load Shed
# ============================================================

def fill_load_shed(df):
    logger.info(
        "Filling missing load-shed values"
    )

    df[
        "Load shed (MW)"
    ] = (
        df[
            "Load shed (MW)"
        ]
        .fillna(0.0)
    )

    return df


# ============================================================
# Validation
# ============================================================

def validate_processed_data(df):
    logger.info(
        "Validating processed dataset"
    )

    duplicate_count = (
        df.duplicated(
            subset=[
                "Date",
                "Zone Name",
            ]
        )
        .sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Processed dataset contains "
            f"{duplicate_count} duplicate rows"
        )

    missing_demand = (
        df["Demand (MW)"]
        .isna()
        .sum()
    )

    missing_load_shed = (
        df["Load shed (MW)"]
        .isna()
        .sum()
    )

    if missing_demand != 0:
        raise ValueError(
            f"Demand still contains "
            f"{missing_demand} missing values"
        )

    if missing_load_shed != 0:
        raise ValueError(
            f"Load shed still contains "
            f"{missing_load_shed} missing values"
        )

    region_counts = (
        df.groupby("Date")[
            "Zone Name"
        ]
        .nunique()
    )

    invalid_dates = region_counts[
        region_counts != len(REGIONS)
    ]

    if not invalid_dates.empty:
        raise ValueError(
            "Some dates do not contain "
            "all expected regions"
        )

    expected_rows = (
        df["Date"].nunique()
        * len(REGIONS)
    )

    if len(df) != expected_rows:
        raise ValueError(
            "Processed row count does not "
            "match complete calendar grid"
        )

    if not set(
        df["is_imputed"].unique()
    ).issubset(
        {0, 1}
    ):
        raise ValueError(
            "Invalid is_imputed values found"
        )

    logger.info(
        "Processed dataset validation passed"
    )


# ============================================================
# Summary
# ============================================================

def print_summary(df):
    total_rows = len(df)

    total_dates = (
        df["Date"].nunique()
    )

    total_regions = (
        df["Zone Name"].nunique()
    )

    imputed_rows = int(
        df["is_imputed"].sum()
    )

    imputed_dates = (
        df.loc[
            df["is_imputed"] == 1,
            "Date",
        ]
        .nunique()
    )

    real_rows = (
        total_rows
        - imputed_rows
    )

    print(
        "\nPREPROCESSING SUMMARY\n"
    )

    print(
        f"Date range:     "
        f"{df['Date'].min().date()} "
        f"to "
        f"{df['Date'].max().date()}"
    )

    print(
        f"Total dates:    "
        f"{total_dates}"
    )

    print(
        f"Regions:        "
        f"{total_regions}"
    )

    print(
        f"Total rows:     "
        f"{total_rows}"
    )

    print(
        f"Real rows:      "
        f"{real_rows}"
    )

    print(
        f"Imputed rows:   "
        f"{imputed_rows}"
    )

    print(
        f"Dates affected "
        f"by imputation: "
        f"{imputed_dates}"
    )

    print(
        f"Latest raw date:"
        f" {df['Date'].max().date()}"
    )


# ============================================================
# Save
# ============================================================

def save_processed_data(df):
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = (
        df
        .sort_values(
            [
                "Date",
                "Zone Name",
            ]
        )
        .reset_index(drop=True)
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    logger.info(
        f"Processed dataset saved: "
        f"{OUTPUT_PATH}"
    )


# ============================================================
# Main
# ============================================================

def main():
    logger.info(
        "Starting BPDB preprocessing"
    )

    df = load_raw_data()

    df = remove_duplicates(
        df
    )

    df = apply_known_corrections(
        df
    )

    df = build_complete_grid(
        df
    )

    df = interpolate_demand(
        df
    )

    df = fill_load_shed(
        df
    )

    validate_processed_data(
        df
    )

    save_processed_data(
        df
    )

    print_summary(
        df
    )

    logger.info(
        "BPDB preprocessing completed successfully"
    )


if __name__ == "__main__":
    main()