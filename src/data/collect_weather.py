import requests
import pandas as pd

from src.logger import get_logger


logger = get_logger("collect_weather")

OUTPUT_PATH = "data/raw/dhaka_weather.csv"

# Dhaka approximate coordinates
LATITUDE = 23.8103
LONGITUDE = 90.4125

START_DATE = "2020-01-01"
END_DATE = "2026-09-24"


def collect_weather():
    logger.info("Starting Dhaka historical weather collection")

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
        ],
        "timezone": "Asia/Dhaka",
    }

    logger.info(
        f"Requesting weather data from "
        f"{START_DATE} to {END_DATE}"
    )

    response = requests.get(
        url,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    if "daily" not in data:
        raise ValueError(
            "Daily weather data not found in API response"
        )

    daily = data["daily"]

    df = pd.DataFrame(
        {
            "Date": daily["time"],
            "temperature_max_c": daily[
                "temperature_2m_max"
            ],
            "temperature_min_c": daily[
                "temperature_2m_min"
            ],
            "temperature_mean_c": daily[
                "temperature_2m_mean"
            ],
            "precipitation_mm": daily[
                "precipitation_sum"
            ],
            "rain_mm": daily[
                "rain_sum"
            ],
        }
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    logger.info(
        f"Collected rows: {len(df)}"
    )

    logger.info(
        f"Date range: "
        f"{df['Date'].min().date()} "
        f"to "
        f"{df['Date'].max().date()}"
    )

    missing_values = (
        df.isna()
        .sum()
        .sum()
    )

    logger.info(
        f"Total missing values: {missing_values}"
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    logger.info(
        f"Weather data saved to: {OUTPUT_PATH}"
    )

    logger.info(
        "Weather collection completed successfully"
    )


if __name__ == "__main__":
    collect_weather()