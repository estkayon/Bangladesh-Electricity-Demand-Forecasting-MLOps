from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from src.logger import get_logger


logger = get_logger("collect_weather")


# ============================================================
# Configuration
# ============================================================

RAW_DEMAND_PATH = Path(
    "data/raw/area_wise_demand.csv"
)

WEATHER_OUTPUT_PATH = Path(
    "data/raw/dhaka_weather.csv"
)


# Dhaka coordinates
LATITUDE = 23.8103
LONGITUDE = 90.4125

TIMEZONE = "Asia/Dhaka"

START_DATE = pd.Timestamp(
    "2020-01-01"
)


ARCHIVE_API_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)

HISTORICAL_FORECAST_API_URL = (
    "https://historical-forecast-api.open-meteo.com/v1/forecast"
)


DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
]


# Open-Meteo archive data can lag behind
# recent dates. We use historical forecast
# API for very recent missing dates.
ARCHIVE_SAFETY_LAG_DAYS = 5


# ============================================================
# Date Helpers
# ============================================================

def get_bangladesh_today():
    return pd.Timestamp.now(
        tz=ZoneInfo(TIMEZONE)
    ).normalize().tz_localize(None)


def get_latest_bpdb_date():
    if not RAW_DEMAND_PATH.exists():
        raise FileNotFoundError(
            f"Raw BPDB dataset not found: "
            f"{RAW_DEMAND_PATH}"
        )

    df = pd.read_csv(
        RAW_DEMAND_PATH,
        usecols=["Date"],
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Date"]
    )

    if df.empty:
        raise ValueError(
            "Raw BPDB dataset contains "
            "no valid dates"
        )

    latest_date = (
        df["Date"]
        .max()
        .normalize()
    )

    logger.info(
        "Latest real BPDB date: "
        f"{latest_date.date()}"
    )

    return latest_date


# ============================================================
# Existing Weather Data
# ============================================================

def load_existing_weather():
    if not WEATHER_OUTPUT_PATH.exists():
        logger.info(
            "No existing weather dataset found"
        )

        return pd.DataFrame(
            columns=[
                "Date",
                "temperature_max_c",
                "temperature_min_c",
                "temperature_mean_c",
                "precipitation_mm",
                "rain_mm",
            ]
        )

    logger.info(
        f"Loading existing weather data: "
        f"{WEATHER_OUTPUT_PATH}"
    )

    df = pd.read_csv(
        WEATHER_OUTPUT_PATH
    )

    required_columns = {
        "Date",
        "temperature_max_c",
        "temperature_min_c",
        "temperature_mean_c",
        "precipitation_mm",
        "rain_mm",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Existing weather dataset is "
            "missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Date"]
    )

    numeric_columns = [
        "temperature_max_c",
        "temperature_min_c",
        "temperature_mean_c",
        "precipitation_mm",
        "rain_mm",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if not df.empty:
        logger.info(
            "Existing weather range: "
            f"{df['Date'].min().date()} "
            "to "
            f"{df['Date'].max().date()}"
        )

        logger.info(
            f"Existing weather rows: "
            f"{len(df)}"
        )

    return df


# ============================================================
# Open-Meteo Request
# ============================================================

def request_weather(
    url,
    start_date,
    end_date,
):
    logger.info(
        f"Requesting weather: "
        f"{start_date.date()} "
        f"to {end_date.date()}"
    )

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": (
            start_date.strftime(
                "%Y-%m-%d"
            )
        ),
        "end_date": (
            end_date.strftime(
                "%Y-%m-%d"
            )
        ),
        "daily": ",".join(
            DAILY_VARIABLES
        ),
        "timezone": TIMEZONE,
    }

    response = requests.get(
        url,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    payload = response.json()

    daily = payload.get(
        "daily"
    )

    if not daily:
        raise ValueError(
            "Open-Meteo response does not "
            "contain daily weather data"
        )

    if not daily.get("time"):
        raise ValueError(
            "Open-Meteo returned no "
            "weather dates"
        )

    weather_df = pd.DataFrame(
        {
            "Date":
                pd.to_datetime(
                    daily["time"]
                ),

            "temperature_max_c":
                daily.get(
                    "temperature_2m_max"
                ),

            "temperature_min_c":
                daily.get(
                    "temperature_2m_min"
                ),

            "temperature_mean_c":
                daily.get(
                    "temperature_2m_mean"
                ),

            "precipitation_mm":
                daily.get(
                    "precipitation_sum"
                ),

            "rain_mm":
                daily.get(
                    "rain_sum"
                ),
        }
    )

    return weather_df


# ============================================================
# Archive Weather
# ============================================================

def collect_archive_weather(
    start_date,
    end_date,
):
    if start_date > end_date:
        return pd.DataFrame()

    logger.info(
        "Collecting weather from "
        "Open-Meteo Archive API"
    )

    return request_weather(
        url=ARCHIVE_API_URL,
        start_date=start_date,
        end_date=end_date,
    )


# ============================================================
# Recent Historical Weather
# ============================================================

def collect_recent_weather(
    start_date,
    end_date,
):
    if start_date > end_date:
        return pd.DataFrame()

    logger.info(
        "Collecting recent weather from "
        "Open-Meteo Historical Forecast API"
    )

    return request_weather(
        url=HISTORICAL_FORECAST_API_URL,
        start_date=start_date,
        end_date=end_date,
    )


# ============================================================
# Missing Date Detection
# ============================================================

def get_missing_dates(
    existing_df,
    target_end_date,
):
    expected_dates = pd.date_range(
        start=START_DATE,
        end=target_end_date,
        freq="D",
    )

    if existing_df.empty:
        return list(
            expected_dates
        )

    existing_dates = set(
        existing_df["Date"]
        .dt.normalize()
    )

    missing_dates = [
        date_value
        for date_value
        in expected_dates
        if date_value
        not in existing_dates
    ]

    return missing_dates


# ============================================================
# Group Consecutive Missing Dates
# ============================================================

def group_consecutive_dates(
    dates,
):
    if not dates:
        return []

    sorted_dates = sorted(
        pd.Timestamp(d).normalize()
        for d in dates
    )

    ranges = []

    range_start = sorted_dates[0]
    previous_date = sorted_dates[0]

    for current_date in (
        sorted_dates[1:]
    ):
        expected_next = (
            previous_date
            + pd.Timedelta(days=1)
        )

        if current_date != expected_next:
            ranges.append(
                (
                    range_start,
                    previous_date,
                )
            )

            range_start = current_date

        previous_date = current_date

    ranges.append(
        (
            range_start,
            previous_date,
        )
    )

    return ranges


# ============================================================
# Collect Missing Weather
# ============================================================

def collect_missing_weather(
    missing_dates,
):
    if not missing_dates:
        logger.info(
            "No missing weather dates found"
        )

        return pd.DataFrame()

    today = get_bangladesh_today()

    archive_cutoff = (
        today
        - pd.Timedelta(
            days=ARCHIVE_SAFETY_LAG_DAYS
        )
    )

    logger.info(
        f"Archive cutoff date: "
        f"{archive_cutoff.date()}"
    )

    collected_frames = []

    missing_ranges = (
        group_consecutive_dates(
            missing_dates
        )
    )

    for (
        start_date,
        end_date,
    ) in missing_ranges:

        # ----------------------------------------------------
        # Entire range old enough for Archive API
        # ----------------------------------------------------

        if end_date <= archive_cutoff:
            try:
                frame = (
                    collect_archive_weather(
                        start_date,
                        end_date,
                    )
                )

                if not frame.empty:
                    collected_frames.append(
                        frame
                    )

                continue

            except Exception as error:
                logger.warning(
                    "Archive API failed for "
                    f"{start_date.date()} "
                    f"to {end_date.date()}: "
                    f"{error}"
                )

                logger.info(
                    "Trying Historical "
                    "Forecast API fallback"
                )

                frame = (
                    collect_recent_weather(
                        start_date,
                        end_date,
                    )
                )

                if not frame.empty:
                    collected_frames.append(
                        frame
                    )

                continue

        # ----------------------------------------------------
        # Range crosses archive cutoff
        # ----------------------------------------------------

        if start_date <= archive_cutoff:
            archive_end = (
                archive_cutoff
            )

            try:
                frame = (
                    collect_archive_weather(
                        start_date,
                        archive_end,
                    )
                )

                if not frame.empty:
                    collected_frames.append(
                        frame
                    )

            except Exception as error:
                logger.warning(
                    "Archive request failed: "
                    f"{error}"
                )

                frame = (
                    collect_recent_weather(
                        start_date,
                        archive_end,
                    )
                )

                if not frame.empty:
                    collected_frames.append(
                        frame
                    )

            recent_start = (
                archive_cutoff
                + pd.Timedelta(days=1)
            )

        else:
            recent_start = (
                start_date
            )

        # ----------------------------------------------------
        # Recent section
        # ----------------------------------------------------

        if recent_start <= end_date:
            frame = (
                collect_recent_weather(
                    recent_start,
                    end_date,
                )
            )

            if not frame.empty:
                collected_frames.append(
                    frame
                )

    if not collected_frames:
        return pd.DataFrame()

    return pd.concat(
        collected_frames,
        ignore_index=True,
    )


# ============================================================
# Merge Weather
# ============================================================

def merge_weather(
    existing_df,
    new_df,
):
    frames = []

    if not existing_df.empty:
        frames.append(
            existing_df
        )

    if not new_df.empty:
        frames.append(
            new_df
        )

    if not frames:
        return pd.DataFrame()

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Date"]
    )

    df = (
        df
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# Validation
# ============================================================

def validate_weather(
    df,
    target_end_date,
):
    logger.info(
        "Validating weather dataset"
    )

    if df.empty:
        raise ValueError(
            "Weather dataset is empty"
        )

    expected_dates = pd.date_range(
        start=START_DATE,
        end=target_end_date,
        freq="D",
    )

    available_dates = set(
        df["Date"]
        .dt.normalize()
    )

    missing_dates = [
        date_value
        for date_value
        in expected_dates
        if date_value
        not in available_dates
    ]

    if missing_dates:
        first_missing = (
            missing_dates[0]
            .date()
        )

        raise ValueError(
            "Weather dataset still has "
            f"{len(missing_dates)} missing "
            "dates. First missing date: "
            f"{first_missing}"
        )

    required_columns = [
        "temperature_max_c",
        "temperature_min_c",
        "temperature_mean_c",
        "precipitation_mm",
        "rain_mm",
    ]

    missing_values = (
        df[
            required_columns
        ]
        .isna()
        .sum()
    )

    problem_columns = (
        missing_values[
            missing_values > 0
        ]
    )

    if not problem_columns.empty:
        logger.warning(
            "Weather dataset contains "
            "missing weather values:\n"
            f"{problem_columns}"
        )

    duplicate_count = (
        df.duplicated(
            subset=["Date"]
        )
        .sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            "Weather dataset contains "
            f"{duplicate_count} duplicate dates"
        )

    latest_weather_date = (
        df["Date"].max()
    )

    if latest_weather_date < target_end_date:
        raise ValueError(
            "Weather data does not reach "
            "latest BPDB date"
        )

    logger.info(
        "Weather validation passed"
    )


# ============================================================
# Save
# ============================================================

def save_weather(
    df,
    target_end_date,
):
    # Keep only the range required by
    # the current demand dataset.
    df = df[
        (
            df["Date"] >= START_DATE
        )
        &
        (
            df["Date"] <= target_end_date
        )
    ].copy()

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    WEATHER_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        WEATHER_OUTPUT_PATH,
        index=False,
    )

    logger.info(
        f"Weather dataset saved: "
        f"{WEATHER_OUTPUT_PATH}"
    )

    return df


# ============================================================
# Summary
# ============================================================

def print_summary(
    df,
    target_end_date,
    newly_collected_rows,
):
    print(
        "\nWEATHER COLLECTION SUMMARY\n"
    )

    print(
        "Target BPDB date: "
        f"{target_end_date.date()}"
    )

    print(
        "Weather range:    "
        f"{df['Date'].min().date()} "
        "to "
        f"{df['Date'].max().date()}"
    )

    print(
        f"Total rows:       "
        f"{len(df)}"
    )

    print(
        f"New rows fetched: "
        f"{newly_collected_rows}"
    )

    print(
        "Weather coverage: COMPLETE"
    )


# ============================================================
# Main
# ============================================================

def main():
    logger.info(
        "Starting dynamic weather collection"
    )

    target_end_date = (
        get_latest_bpdb_date()
    )

    existing_df = (
        load_existing_weather()
    )

    missing_dates = (
        get_missing_dates(
            existing_df,
            target_end_date,
        )
    )

    logger.info(
        f"Missing weather dates: "
        f"{len(missing_dates)}"
    )

    new_df = (
        collect_missing_weather(
            missing_dates
        )
    )

    newly_collected_rows = (
        len(new_df)
    )

    combined_df = (
        merge_weather(
            existing_df,
            new_df,
        )
    )

    validate_weather(
        combined_df,
        target_end_date,
    )

    final_df = save_weather(
        combined_df,
        target_end_date,
    )

    print_summary(
        final_df,
        target_end_date,
        newly_collected_rows,
    )

    logger.info(
        "Dynamic weather collection "
        "completed successfully"
    )


if __name__ == "__main__":
    main()