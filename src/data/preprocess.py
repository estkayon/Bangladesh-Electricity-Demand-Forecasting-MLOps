import pandas as pd

from src.logger import get_logger

logger = get_logger("preprocess")

RAW_PATH = "data/raw/area_wise_demand.csv"
OUTPUT_PATH = "data/processed/area_wise_demand_processed.csv"

df = pd.read_csv(RAW_PATH)

# Convert date column
df["Date"] = pd.to_datetime(df["Date"])

# Keep original rows marked as non-imputed
df["is_imputed"] = 0

# Get all zones
zones = sorted(df["Zone Name"].unique())

# Create complete date range
full_dates = pd.date_range(
    start="2020-01-01",
    end="2026-09-24",
    freq="D"
)

# Create full Date × Zone grid
full_index = pd.MultiIndex.from_product(
    [full_dates, zones],
    names=["Date", "Zone Name"]
)

df = (
    df.set_index(["Date", "Zone Name"])
      .reindex(full_index)
      .reset_index()
)

# Mark missing rows
df["is_imputed"] = df["is_imputed"].fillna(1).astype(int)

# Sort properly before interpolation
df = df.sort_values(["Zone Name", "Date"])

# Interpolate demand separately for each zone
df["Demand (MW)"] = (
    df.groupby("Zone Name")["Demand (MW)"]
      .transform(lambda x: x.interpolate(method="linear"))
)

df["Load shed (MW)"] = df["Load shed (MW)"].fillna(0)

df["is_anomaly"] = 0

demand_anomalies = (
    (df["Zone Name"] == "Mymensingh") & (df["Date"] == "2026-08-10")
) | (
    (df["Zone Name"] == "Comilla") & (df["Date"] == "2023-11-02")
) | (
    (df["Zone Name"] == "Khulna") & (df["Date"] == "2024-08-01")
)

load_shed_anomalies = (
    (df["Zone Name"] == "Khulna") & (df["Date"] == "2023-01-06")
)

df.loc[demand_anomalies | load_shed_anomalies, "is_anomaly"] = 1

logger.info("Replacing flagged anomaly values")

# Demand anomalies
demand_anomaly_mask = demand_anomalies

for idx in df[demand_anomaly_mask].index:
    zone = df.loc[idx, "Zone Name"]
    date = df.loc[idx, "Date"]

    prev_value = df[
        (df["Zone Name"] == zone) &
        (df["Date"] == date - pd.Timedelta(days=1))
    ]["Demand (MW)"].values

    next_value = df[
        (df["Zone Name"] == zone) &
        (df["Date"] == date + pd.Timedelta(days=1))
    ]["Demand (MW)"].values

    if len(prev_value) > 0 and len(next_value) > 0:
        df.loc[idx, "Demand (MW)"] = (
            prev_value[0] + next_value[0]
        ) / 2


# Load shedding anomaly
for idx in df[load_shed_anomalies].index:
    zone = df.loc[idx, "Zone Name"]
    date = df.loc[idx, "Date"]

    prev_value = df[
        (df["Zone Name"] == zone) &
        (df["Date"] == date - pd.Timedelta(days=1))
    ]["Load shed (MW)"].values

    next_value = df[
        (df["Zone Name"] == zone) &
        (df["Date"] == date + pd.Timedelta(days=1))
    ]["Load shed (MW)"].values

    if len(prev_value) > 0 and len(next_value) > 0:
        df.loc[idx, "Load shed (MW)"] = (
            prev_value[0] + next_value[0]
        ) / 2

logger.info("Anomaly replacement completed")


# Save processed dataset
df.to_csv(OUTPUT_PATH, index=False)

print(f"Processed rows: {len(df)}")
print(f"Imputed rows: {df['is_imputed'].sum()}")
print(f"Saved to: {OUTPUT_PATH}")

print("\nMissing Demand values:", df["Demand (MW)"].isna().sum())
print("Missing Load Shed values:", df["Load shed (MW)"].isna().sum())

print("\nSample missing load-shed rows:")
print(
    df[df["Load shed (MW)"].isna()]
    [["Date", "Zone Name", "Load shed (MW)", "is_imputed"]]
    .head(20)
)

print("\nDuplicate rows:",
      df.duplicated(subset=["Date", "Zone Name"]).sum())

print("Unique zones:",
      df["Zone Name"].nunique())

print("Date range:",
      df["Date"].min(),
      "to",
      df["Date"].max())

print("Rows per date min/max:")
print(
    df.groupby("Date")["Zone Name"]
      .count()
      .agg(["min", "max"])
)

logger.info(f"Anomaly rows flagged: {df['is_anomaly'].sum()}")

