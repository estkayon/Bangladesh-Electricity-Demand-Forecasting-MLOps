import time
import requests
import pandas as pd
import urllib3

from io import StringIO

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def collect_single_date(date_str):
    url = f"https://misc.bpdb.gov.bd/area-wise-demand?date={date_str}"

    response = requests.get(url, timeout=30, verify=False)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))

    if len(tables) < 2:
        raise ValueError("Demand table not found")

    table = tables[1].copy()
    table = table[table["Zone Name"] != "Total"]

    table["Demand (MW)"] = pd.to_numeric(
        table["Demand (MW)"],
        errors="coerce"
    )

    table["Load shed (MW)"] = pd.to_numeric(
        table["Load shed (MW)"],
        errors="coerce"
    )

    parsed_date = pd.to_datetime(
        date_str,
        format="%d-%m-%Y"
    ).strftime("%Y-%m-%d")

    table["Date"] = parsed_date

    table = table[
        ["Date", "Zone Name", "Demand (MW)", "Load shed (MW)"]
    ]

    return table


missing_df = pd.read_csv("data/raw/missing_dates.csv")

recovered_data = []
still_failed = []

for date_str in missing_df["missing_date"]:
    print(f"Retrying: {date_str}")

    try:
        daily_data = collect_single_date(date_str)
        recovered_data.append(daily_data)

        print("Recovered")

    except Exception as e:
        print(f"Still failed: {date_str} | {e}")
        still_failed.append(date_str)

    time.sleep(1)