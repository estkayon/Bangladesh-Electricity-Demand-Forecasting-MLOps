from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup

from src.logger import get_logger


logger = get_logger("collect_data")


# ============================================================
# Configuration
# ============================================================

BASE_URL = (
    "https://misc.bpdb.gov.bd/area-wise-demand"
)

RAW_DATA_PATH = Path(
    "data/raw/area_wise_demand.csv"
)

MISSING_DATES_PATH = Path(
    "data/raw/missing_dates.csv"
)

TIMEZONE = "Asia/Dhaka"

REQUEST_TIMEOUT = 30


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


# BPDB certificate currently causes
# verification issues in some environments.
urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)


# ============================================================
# Date Helpers
# ============================================================

def get_bangladesh_today():
    return (
        pd.Timestamp.now(
            tz=ZoneInfo(TIMEZONE)
        )
        .normalize()
        .tz_localize(None)
    )


# ============================================================
# Existing Raw Data
# ============================================================

def load_existing_raw_data():
    if not RAW_DATA_PATH.exists():
        logger.info(
            "No existing raw BPDB dataset found"
        )

        return pd.DataFrame(
            columns=[
                "Date",
                "Zone Name",
                "Demand (MW)",
                "Load shed (MW)",
            ]
        )

    logger.info(
        f"Loading existing raw data: "
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
            "Raw dataset missing columns: "
            f"{sorted(missing_columns)}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df["Zone Name"] = (
        df["Zone Name"]
        .astype(str)
        .str.strip()
    )

    df["Demand (MW)"] = pd.to_numeric(
        df["Demand (MW)"],
        errors="coerce",
    )

    df["Load shed (MW)"] = pd.to_numeric(
        df["Load shed (MW)"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Date",
            "Zone Name",
            "Demand (MW)",
        ]
    ).copy()

    df = df[
        df["Zone Name"].isin(
            REGIONS
        )
    ].copy()

    df = (
        df
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

    logger.info(
        f"Existing raw rows: "
        f"{len(df)}"
    )

    if not df.empty:
        logger.info(
            "Existing raw range: "
            f"{df['Date'].min().date()} "
            "to "
            f"{df['Date'].max().date()}"
        )

    return df


# ============================================================
# Existing Missing-Date Tracking
# ============================================================

def load_existing_missing_dates():
    if not MISSING_DATES_PATH.exists():
        logger.info(
            "No existing missing-date file found"
        )

        return pd.DataFrame(
            columns=[
                "Date",
                "status",
            ]
        )

    logger.info(
        f"Loading existing missing dates: "
        f"{MISSING_DATES_PATH}"
    )

    df = pd.read_csv(
        MISSING_DATES_PATH
    )

    if "Date" not in df.columns:
        raise ValueError(
            "missing_dates.csv must "
            "contain a Date column"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Date"]
    ).copy()

    if "status" not in df.columns:
        df["status"] = "missing"

    df = (
        df
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    logger.info(
        f"Existing tracked missing dates: "
        f"{len(df)}"
    )

    return df


# ============================================================
# HTML Parsing
# ============================================================

def clean_numeric_value(value):
    if value is None:
        return None

    value = (
        str(value)
        .replace(",", "")
        .strip()
    )

    if value == "":
        return None

    try:
        return float(value)

    except ValueError:
        return None


def parse_bpdb_table(
    html,
    requested_date,
):
    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    tables = soup.find_all(
        "table"
    )

    if not tables:
        return None

    valid_rows = []

    for table in tables:
        rows = table.find_all(
            "tr"
        )

        for row in rows:
            cells = [
                cell.get_text(
                    " ",
                    strip=True,
                )
                for cell in row.find_all(
                    [
                        "td",
                        "th",
                    ]
                )
            ]

            if len(cells) < 3:
                continue

            zone_name = cells[0].strip()

            if zone_name not in REGIONS:
                continue

            demand_value = (
                clean_numeric_value(
                    cells[1]
                )
            )

            load_shed_value = (
                clean_numeric_value(
                    cells[2]
                )
            )

            if demand_value is None:
                continue

            if load_shed_value is None:
                load_shed_value = 0.0

            valid_rows.append(
                {
                    "Date":
                        requested_date,

                    "Zone Name":
                        zone_name,

                    "Demand (MW)":
                        demand_value,

                    "Load shed (MW)":
                        load_shed_value,
                }
            )

    if not valid_rows:
        return None

    result_df = pd.DataFrame(
        valid_rows
    )

    result_df = (
        result_df
        .drop_duplicates(
            subset=[
                "Date",
                "Zone Name",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    found_regions = set(
        result_df[
            "Zone Name"
        ]
    )

    expected_regions = set(
        REGIONS
    )

    if found_regions != expected_regions:
        logger.warning(
            f"{requested_date.date()} "
            "did not contain all 9 regions. "
            f"Found: {sorted(found_regions)}"
        )

        return None

    return result_df


# ============================================================
# Fetch One Date
# ============================================================

def fetch_date(
    target_date,
):
    date_text = (
        target_date.strftime(
            "%d-%m-%Y"
        )
    )

    logger.info(
        f"Fetching BPDB data: "
        f"{date_text}"
    )

    try:
        response = requests.get(
            BASE_URL,
            params={
                "date":
                    date_text
            },
            timeout=REQUEST_TIMEOUT,
            verify=False,
        )

        response.raise_for_status()

        parsed_df = (
            parse_bpdb_table(
                response.text,
                target_date,
            )
        )

        if (
            parsed_df is None
            or parsed_df.empty
        ):
            logger.warning(
                f"No valid demand table: "
                f"{date_text}"
            )

            return None

        logger.info(
            f"Valid BPDB data found: "
            f"{date_text}"
        )

        return parsed_df

    except requests.RequestException as error:
        logger.warning(
            f"Request failed for "
            f"{date_text}: {error}"
        )

        return None


# ============================================================
# Determine Dates to Probe
# ============================================================
def determine_dates_to_probe(
    raw_df,
    missing_df,
):
    today = get_bangladesh_today()

    dates_to_probe = set()

    # --------------------------------------------------------
    # Validate raw dataset
    # --------------------------------------------------------

    if raw_df.empty:
        raise ValueError(
            "Raw dataset is empty. "
            "Initial full historical "
            "collection is required."
        )

    latest_real_date = (
        raw_df["Date"]
        .max()
        .normalize()
    )

    # --------------------------------------------------------
    # Retry only recent not-yet-available dates
    # --------------------------------------------------------

    if not missing_df.empty:
        recent_missing = missing_df.copy()

        if "status" in recent_missing.columns:
            recent_missing = recent_missing[
                recent_missing["status"]
                == "not_yet_available"
            ]

        for missing_date in recent_missing["Date"]:
            if (
                pd.notna(missing_date)
                and latest_real_date
                < missing_date
                <= today
            ):
                dates_to_probe.add(
                    missing_date.normalize()
                )

    # --------------------------------------------------------
    # Probe forward from latest real raw date
    # --------------------------------------------------------

    next_date = (
        latest_real_date
        + pd.Timedelta(days=1)
    )

    if next_date <= today:
        forward_dates = pd.date_range(
            start=next_date,
            end=today,
            freq="D",
        )

        for target_date in forward_dates:
            dates_to_probe.add(
                target_date.normalize()
            )

    return sorted(
        dates_to_probe
    )

# ============================================================
# Merge Raw Data
# ============================================================

def merge_raw_data(
    existing_df,
    new_frames,
):
    frames = [
        existing_df
    ]

    frames.extend(
        new_frames
    )

    combined_df = pd.concat(
        frames,
        ignore_index=True,
    )

    combined_df["Date"] = (
        pd.to_datetime(
            combined_df["Date"]
        )
    )

    combined_df = (
        combined_df
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

    return combined_df


# ============================================================
# Rebuild Missing-Date List
# ============================================================

def rebuild_missing_dates(
    raw_df,
    existing_missing_df,
    probed_dates,
):
    # --------------------------------------------------------
    # Historical calendar gaps from first raw date
    # through latest real raw date
    # --------------------------------------------------------

    raw_dates = set(
        raw_df[
            "Date"
        ]
        .dt.normalize()
        .unique()
    )

    first_raw_date = (
        raw_df[
            "Date"
        ]
        .min()
        .normalize()
    )

    latest_real_date = (
        raw_df[
            "Date"
        ]
        .max()
        .normalize()
    )

    complete_calendar = pd.date_range(
        start=first_raw_date,
        end=latest_real_date,
        freq="D",
    )

    historical_missing = {
        target_date
        for target_date
        in complete_calendar
        if target_date
        not in raw_dates
    }

    # --------------------------------------------------------
    # Preserve current recent unavailable dates
    # beyond latest real date
    # --------------------------------------------------------

    existing_missing_df["Date"] = pd.to_datetime(
        existing_missing_df["Date"],
        errors="coerce",
    )

    previously_tracked = set(
        existing_missing_df[
            "Date"
        ]
        .dropna()
        .dt.normalize()
    )

    probed_set = set(
        pd.Timestamp(date_value)
        .normalize()
        for date_value
        in probed_dates
    )

    # Any probed date which still has
    # no raw data remains unavailable.
    still_unavailable = {
        target_date
        for target_date
        in probed_set
        if target_date
        not in raw_dates
    }

    # Preserve tracked future/recent unavailable
    # dates unless they are now available.
    preserved_unavailable = {
        target_date
        for target_date
        in previously_tracked
        if target_date
        not in raw_dates
    }

    all_missing_dates = (
        historical_missing
        | still_unavailable
        | preserved_unavailable
    )

    rows = []

    for target_date in sorted(
        all_missing_dates
    ):
        if target_date <= latest_real_date:
            status = (
                "historical_missing"
            )

        else:
            status = (
                "not_yet_available"
            )

        rows.append(
            {
                "Date":
                    target_date,

                "status":
                    status,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Validation
# ============================================================

def validate_raw_data(
    df,
):
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
            f"Raw dataset contains "
            f"{duplicate_count} duplicate rows"
        )

    invalid_regions = set(
        df["Zone Name"].unique()
    ) - set(REGIONS)

    if invalid_regions:
        raise ValueError(
            "Unexpected regions found: "
            f"{sorted(invalid_regions)}"
        )

    per_date_count = (
        df.groupby("Date")[
            "Zone Name"
        ]
        .nunique()
    )

    invalid_dates = (
        per_date_count[
            per_date_count
            != len(REGIONS)
        ]
    )

    if not invalid_dates.empty:
        raise ValueError(
            "Some stored BPDB dates "
            "do not contain all 9 regions"
        )


# ============================================================
# Save
# ============================================================

def save_raw_data(
    df,
):
    RAW_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df = df.copy()

    output_df["Date"] = (
        output_df["Date"]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    output_df.to_csv(
        RAW_DATA_PATH,
        index=False,
    )

    logger.info(
        f"Raw dataset saved: "
        f"{RAW_DATA_PATH}"
    )


def save_missing_dates(
    df,
):
    MISSING_DATES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if df.empty:
        empty_df = pd.DataFrame(
            columns=[
                "Date",
                "status",
            ]
        )

        empty_df.to_csv(
            MISSING_DATES_PATH,
            index=False,
        )

        logger.info(
            "No missing dates remain"
        )

        return

    output_df = df.copy()

    output_df["Date"] = (
        output_df["Date"]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    output_df.to_csv(
        MISSING_DATES_PATH,
        index=False,
    )

    logger.info(
        f"Missing dates saved: "
        f"{len(output_df)}"
    )


# ============================================================
# Summary
# ============================================================

def print_summary(
    raw_df,
    missing_df,
    dates_to_probe,
    successful_dates,
):
    latest_real_date = (
        raw_df[
            "Date"
        ]
        .max()
    )

    historical_missing = (
        missing_df[
            missing_df[
                "status"
            ]
            == "historical_missing"
        ]
        if not missing_df.empty
        else pd.DataFrame()
    )

    unavailable_dates = (
        missing_df[
            missing_df[
                "status"
            ]
            == "not_yet_available"
        ]
        if not missing_df.empty
        else pd.DataFrame()
    )

    print(
        "\nBPDB COLLECTION SUMMARY\n"
    )

    print(
        f"Dates probed:              "
        f"{len(dates_to_probe)}"
    )

    print(
        f"New/recovered dates found: "
        f"{len(successful_dates)}"
    )

    print(
        f"Raw rows:                  "
        f"{len(raw_df)}"
    )

    print(
        f"Real BPDB dates:            "
        f"{raw_df['Date'].nunique()}"
    )

    print(
        f"Latest real BPDB date:      "
        f"{latest_real_date.date()}"
    )

    print(
        f"Historical missing dates:   "
        f"{len(historical_missing)}"
    )

    print(
        f"Not-yet-available dates:    "
        f"{len(unavailable_dates)}"
    )

    print(
        f"Total tracked missing:      "
        f"{len(missing_df)}"
    )


# ============================================================
# Main
# ============================================================

def main():
    logger.info(
        "Starting incremental BPDB collection"
    )

    raw_df = (
        load_existing_raw_data()
    )

    missing_df = (
        load_existing_missing_dates()
    )

    dates_to_probe = (
        determine_dates_to_probe(
            raw_df,
            missing_df,
        )
    )

    logger.info(
        f"Dates to probe: "
        f"{len(dates_to_probe)}"
    )

    new_frames = []
    successful_dates = []

    for target_date in dates_to_probe:
        fetched_df = (
            fetch_date(
                target_date
            )
        )

        if fetched_df is None:
            continue

        new_frames.append(
            fetched_df
        )

        successful_dates.append(
            target_date
        )

    updated_raw_df = (
        merge_raw_data(
            raw_df,
            new_frames,
        )
    )

    validate_raw_data(
        updated_raw_df
    )

    updated_missing_df = (
        rebuild_missing_dates(
            raw_df=
                updated_raw_df,

            existing_missing_df=
                missing_df,

            probed_dates=
                dates_to_probe,
        )
    )

    save_raw_data(
        updated_raw_df
    )

    save_missing_dates(
        updated_missing_df
    )

    print_summary(
        raw_df=
            updated_raw_df,

        missing_df=
            updated_missing_df,

        dates_to_probe=
            dates_to_probe,

        successful_dates=
            successful_dates,
    )

    logger.info(
        "Incremental BPDB collection "
        "completed successfully"
    )


if __name__ == "__main__":
    main()