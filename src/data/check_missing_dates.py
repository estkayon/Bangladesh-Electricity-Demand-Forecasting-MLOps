import pandas as pd

df = pd.read_csv("data/raw/area_wise_demand.csv")

df["Date"] = pd.to_datetime(df["Date"])

expected_dates = pd.date_range(
    start="2020-01-01",
    end="2026-09-24",
    freq="D"
)

available_dates = pd.DatetimeIndex(df["Date"].unique())

missing_dates = expected_dates.difference(available_dates)

print(f"Total expected dates: {len(expected_dates)}")
print(f"Available dates: {len(available_dates)}")
print(f"Missing dates: {len(missing_dates)}")

missing_df = pd.DataFrame({
    "missing_date": missing_dates.strftime("%d-%m-%Y")
})

missing_df.to_csv(
    "data/raw/missing_dates.csv",
    index=False
)

print("Saved to: data/raw/missing_dates.csv")