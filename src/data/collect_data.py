from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import urllib3

from src.logger import get_logger


logger = get_logger("collect_data")


# ============================================================
# Configuration
# ============================================================

BASE_URL = "https://misc.bpdb.gov.bd/area-wise-demand"

RAW_DATA_PATH = Path(
    "data/raw/area_wise_demand.csv"
)

MISSING_DATES_PATH = Path(
    "data/raw/missing_dates.csv"
)

BANGLADESH_TIMEZONE = ZoneInfo(
    "Asia/Dhaka"
)

REQUEST_TIMEOUT = 30


EXPECTED_COLUMNS = [
    "Date",
    "Zone Name",
    "Demand (MW)",
    "Load shed (MW)",
]


# SSL verification previously failed for BPDB.
urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)


# ============================================================
# Utilities
# ============================================================

def get_today_bangladesh():
    return datetime.now(
        BANGLADESH_TIMEZONE
    ).date()


def normalize_column_name(column):
    return (
        str(column)
        .strip()
        .replace("\n", " ")
        .replace("\r", " ")
    )


def normalize_table_columns(df):
    df.columns = [
        normalize_column_name(column)
        for column in df.columns
    ]

    rename_map = {}

    for column in df.columns:
        lower = column.lower()

        if (
            "zone" in lower
            and "name" in lower
        ):
            rename_map[column] = "Zone Name"

        elif (
            "demand" in lower
            and "load" not in lower
        ):
            rename_map[column] = "Demand (MW)"

        elif (
            "load" in lower
            and "shed" in lower
        ):
            rename_map[column] = "Load shed (MW)"

    return df.rename(
        columns=rename_map
    )


def load_existing_data():
    if not RAW_DATA_PATH.exists():
        logger.info(
            "Raw dataset does not exist yet"
        )

        return pd.DataFrame(
            columns=EXPECTED_COLUMNS
        )

    logger.info(
        f"Loading existing raw data: "
        f"{RAW_DATA_PATH}"
    )

    df = pd.read_csv(
        RAW_DATA_PATH
    )

    if df.empty:
        return pd.DataFrame(
            columns=EXPECTED_COLUMNS
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Date"]
    )

    logger.info(
        f"Existing raw rows: {len(df)}"
    )

    logger.info(
        "Existing date range: "
        f"{df['Date'].min().date()} "
        "to "
        f"{df['Date'].max().date()}"
    )

    return df


# ============================================================
# BPDB Scraping
# ============================================================

def fetch_date_data(target_date):
    formatted_date = target_date.strftime(
        "%d-%m-%Y"
    )

    logger.info(
        f"Fetching BPDB data for "
        f"{target_date}"
    )

    try:
        response = requests.get(
            BASE_URL,
            params={
                "date": formatted_date
            },
            timeout=REQUEST_TIMEOUT,
            verify=False,
        )

        response.raise_for_status()

    except requests.RequestException as error:
        logger.warning(
            f"Request failed for "
            f"{target_date}: {error}"
        )

        return None

    try:
        tables = pd.read_html(
            StringIO(
                response.text
            )
        )

    except ValueError:
        logger.warning(
            f"No HTML tables found for "
            f"{target_date}"
        )

        return None

    except Exception as error:
        logger.warning(
            f"Could not parse tables for "
            f"{target_date}: {error}"
        )

        return None

    demand_table = None

    for table in tables:
        table = normalize_table_columns(
            table
        )

        required = {
            "Zone Name",
            "Demand (MW)",
            "Load shed (MW)",
        }

        if required.issubset(
            table.columns
        ):
            demand_table = table.copy()
            break

    if demand_table is None:
        logger.warning(
            f"Demand table not found for "
            f"{target_date}"
        )

        return None

    demand_table = demand_table[
        [
            "Zone Name",
            "Demand (MW)",
            "Load shed (MW)",
        ]
    ].copy()

    # --------------------------------------------------------
    # Clean zone names
    # --------------------------------------------------------

    demand_table[
        "Zone Name"
    ] = (
        demand_table[
            "Zone Name"
        ]
        .astype(str)
        .str.strip()
    )

    # Remove total/footer/invalid rows
    demand_table = demand_table[
        ~demand_table[
            "Zone Name"
        ]
        .str.lower()
        .isin(
            [
                "total",
                "nan",
                "",
            ]
        )
    ].copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in [
        "Demand (MW)",
        "Load shed (MW)",
    ]:
        demand_table[column] = (
            demand_table[column]
            .astype(str)
            .str.replace(
                ",",
                "",
                regex=False,
            )
            .str.strip()
        )

        demand_table[column] = (
            pd.to_numeric(
                demand_table[column],
                errors="coerce",
            )
        )

    demand_table = demand_table.dropna(
        subset=[
            "Zone Name",
            "Demand (MW)",
        ]
    )

    # If load shed is missing in a valid row,
    # treat it as zero.
    demand_table[
        "Load shed (MW)"
    ] = (
        demand_table[
            "Load shed (MW)"
        ]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    expected_region_count = 9

    if (
        demand_table[
            "Zone Name"
        ]
        .nunique()
        != expected_region_count
    ):
        logger.warning(
            f"Unexpected region count for "
            f"{target_date}: "
            f"{demand_table['Zone Name'].nunique()}"
        )

        return None

    if (
        demand_table[
            "Demand (MW)"
        ]
        <= 0
    ).any():
        logger.warning(
            f"Invalid demand values found for "
            f"{target_date}"
        )

        return None

    demand_table.insert(
        0,
        "Date",
        pd.Timestamp(
            target_date
        ),
    )

    logger.info(
        f"Valid BPDB data found for "
        f"{target_date}: "
        f"{len(demand_table)} rows"
    )

    return demand_table


# ============================================================
# Missing-Date Tracking
# ============================================================

def load_missing_dates():
    if not MISSING_DATES_PATH.exists():
        return set()

    try:
        missing_df = pd.read_csv(
            MISSING_DATES_PATH
        )

        if (
            "Date"
            not in missing_df.columns
        ):
            return set()

        dates = pd.to_datetime(
            missing_df["Date"],
            errors="coerce",
        ).dropna()

        return {
            timestamp.date()
            for timestamp in dates
        }

    except Exception:
        return set()


def save_missing_dates(
    missing_dates,
):
    missing_dates = sorted(
        set(missing_dates)
    )

    missing_df = pd.DataFrame(
        {
            "Date": [
                date_value.isoformat()
                for date_value
                in missing_dates
            ]
        }
    )

    MISSING_DATES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    missing_df.to_csv(
        MISSING_DATES_PATH,
        index=False,
    )

    logger.info(
        f"Missing dates saved: "
        f"{len(missing_df)}"
    )


# ============================================================
# Incremental Refresh
# ============================================================

def build_dates_to_fetch(
    existing_df,
):
    today = get_today_bangladesh()

    if existing_df.empty:
        raise ValueError(
            "Existing raw dataset is empty. "
            "Use the historical collection "
            "workflow for the initial dataset."
        )

    latest_existing_date = (
        existing_df[
            "Date"
        ]
        .max()
        .date()
    )

    start_date = (
        latest_existing_date
        + timedelta(days=1)
    )

    logger.info(
        f"Latest real raw date: "
        f"{latest_existing_date}"
    )

    logger.info(
        f"Bangladesh current date: "
        f"{today}"
    )

    if start_date > today:
        return []

    dates = []

    current_date = start_date

    while current_date <= today:
        dates.append(
            current_date
        )

        current_date += timedelta(
            days=1
        )

    return dates


def merge_new_data(
    existing_df,
    new_frames,
):
    if not new_frames:
        return existing_df.copy()

    new_df = pd.concat(
        new_frames,
        ignore_index=True,
    )

    combined = pd.concat(
        [
            existing_df,
            new_df,
        ],
        ignore_index=True,
    )

    combined["Date"] = pd.to_datetime(
        combined["Date"]
    )

    combined = (
        combined
        .drop_duplicates(
            subset=[
                "Date",
                "Zone Name",
            ],
            keep="last",
        )
        .sort_values(
            [
                "Date",
                "Zone Name",
            ]
        )
        .reset_index(drop=True)
    )

    return combined


# ============================================================
# Main
# ============================================================

def main():
    logger.info(
        "Starting incremental BPDB "
        "data refresh"
    )

    RAW_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_df = load_existing_data()

    dates_to_fetch = (
        build_dates_to_fetch(
            existing_df
        )
    )

    if not dates_to_fetch:
        logger.info(
            "No new dates need to be fetched"
        )

        print(
            "\nBPDB DATA REFRESH STATUS\n"
        )

        print(
            "No new calendar dates "
            "need to be checked."
        )

        return

    logger.info(
        f"Dates to check: "
        f"{len(dates_to_fetch)}"
    )

    logger.info(
        "Refresh range: "
        f"{dates_to_fetch[0]} "
        "to "
        f"{dates_to_fetch[-1]}"
    )

    existing_missing = (
        load_missing_dates()
    )

    successful_frames = []

    newly_missing = []

    successful_dates = []

    # --------------------------------------------------------
    # Fetch dates one by one
    # --------------------------------------------------------

    for target_date in dates_to_fetch:
        date_df = fetch_date_data(
            target_date
        )

        if date_df is None:
            newly_missing.append(
                target_date
            )

            continue

        successful_frames.append(
            date_df
        )

        successful_dates.append(
            target_date
        )

        # If previously marked missing,
        # remove it after successful fetch.
        existing_missing.discard(
            target_date
        )

    # --------------------------------------------------------
    # Merge and save raw dataset
    # --------------------------------------------------------

    combined_df = merge_new_data(
        existing_df,
        successful_frames,
    )

    combined_df.to_csv(
        RAW_DATA_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    # --------------------------------------------------------
    # Missing dates
    # --------------------------------------------------------

    updated_missing = (
        existing_missing
        .union(
            set(newly_missing)
        )
    )

    save_missing_dates(
        updated_missing
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    previous_rows = len(
        existing_df
    )

    current_rows = len(
        combined_df
    )

    added_rows = (
        current_rows
        - previous_rows
    )

    print(
        "\nBPDB DATA REFRESH STATUS\n"
    )

    print(
        "Previous raw rows:",
        previous_rows,
    )

    print(
        "Current raw rows:",
        current_rows,
    )

    print(
        "New rows added:",
        added_rows,
    )

    print(
        "Dates checked:",
        len(dates_to_fetch),
    )

    print(
        "Successful dates:",
        len(successful_dates),
    )

    print(
        "Unavailable dates:",
        len(newly_missing),
    )

    if successful_dates:
        print(
            "Newest real BPDB date:",
            max(
                successful_dates
            ),
        )

    else:
        latest_existing = (
            combined_df[
                "Date"
            ]
            .max()
            .date()
        )

        print(
            "Newest real BPDB date:",
            latest_existing,
        )

    if newly_missing:
        print(
            "\nUnavailable dates:"
        )

        for missing_date in (
            newly_missing
        ):
            print(
                f"  {missing_date}"
            )

    logger.info(
        "Incremental BPDB data refresh "
        "completed successfully"
    )


if __name__ == "__main__":
    main()