import requests
import pandas as pd
import urllib3
import time

from io import StringIO

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def collect_single_date(date_str):
    url = f"https://misc.bpdb.gov.bd/area-wise-demand?date={date_str}"

    response = requests.get(url, timeout=30, verify=False)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))

    table = tables[1].copy()
    table = table[table["Zone Name"] != "Total"]

    table["Demand (MW)"] = pd.to_numeric(
        table["Demand (MW)"], errors="coerce"
    )

    table["Load shed (MW)"] = pd.to_numeric(
        table["Load shed (MW)"], errors="coerce"
    )

    parsed_date = pd.to_datetime(date_str, format="%d-%m-%Y").strftime("%Y-%m-%d")
    table["Date"] = parsed_date

    table = table[
        ["Date", "Zone Name", "Demand (MW)", "Load shed (MW)"]
    ]

    return table


date_range = pd.date_range(
    start="2020-01-01",
    end="2026-09-24",
    freq="D"
)

all_data = []

failed_dates = []

for date in date_range:
    date_str = date.strftime("%d-%m-%Y")

    print(f"Collecting: {date_str}")

    try:
        daily_data = collect_single_date(date_str)
        all_data.append(daily_data)

    except Exception as e:
        print(f"Failed: {date_str} | {e}")
        failed_dates.append(date_str)

    time.sleep(0.5)

final_data = pd.concat(all_data, ignore_index=True)

print(final_data)
print(f"\nTotal rows: {len(final_data)}")


output_path = "data/raw/area_wise_demand.csv"

final_data.to_csv(output_path, index=False)

print(f"\nSaved to: {output_path}")

print(f"\nSuccessful rows: {len(final_data)}")
print(f"Failed dates: {len(failed_dates)}")

if failed_dates:
    pd.DataFrame({"failed_date": failed_dates}).to_csv(
        "data/raw/failed_dates.csv",
        index=False
    )

    print("Failed dates saved to: data/raw/failed_dates.csv")