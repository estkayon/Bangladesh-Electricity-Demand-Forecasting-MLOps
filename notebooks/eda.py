import pandas as pd

from src.logger import get_logger

logger = get_logger("eda")

DATA_PATH = "data/processed/national_daily_demand.csv"

logger.info("Loading national daily demand dataset")

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

logger.info(f"Rows: {len(df)}")
logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

print("\nSummary Statistics:")
print(df[["total_demand", "total_load_shed", "next_day_total_demand"]].describe())

logger.info("Checking extreme demand values")

print("\nTop 10 Total Demand Dates:")
print(
    df.nlargest(10, "total_demand")[
        ["Date", "total_demand", "total_load_shed"]
    ]
)

print("\nTop 10 Load Shedding Dates:")
print(
    df.nlargest(10, "total_load_shed")[
        ["Date", "total_demand", "total_load_shed"]
    ]
)

logger.info("Inspecting suspicious dates at zone level")

area_df = pd.read_csv(
    "data/processed/area_wise_demand_processed.csv"
)

area_df["Date"] = pd.to_datetime(area_df["Date"])

suspicious_dates = [
    "2026-08-10",
    "2023-11-02",
    "2024-08-01",
    "2023-01-06"
]

for date in suspicious_dates:
    print(f"\n--- {date} ---")

    print(
        area_df[
            area_df["Date"] == date
        ][
            [
                "Zone Name",
                "Demand (MW)",
                "Load shed (MW)",
                "is_imputed"
            ]
        ]
    )

    logger.info("Checking final dates for imputation")

print("\nFinal 7 days zone-level data:")

print(
    area_df[
        area_df["Date"] >= "2026-09-18"
    ][
        [
            "Date",
            "Zone Name",
            "Demand (MW)",
            "is_imputed"
        ]
    ].to_string(index=False)
)

print("\nDemand by split:")

splits = {
    "Train": df[df["Date"] <= "2024-12-31"],
    "Validation": df[
        (df["Date"] >= "2025-01-01")
        & (df["Date"] <= "2025-12-31")
    ],
    "Test": df[df["Date"] >= "2026-01-01"],
}

for name, split in splits.items():
    print(f"\n{name}")
    print(f"Mean: {split['total_demand'].mean():.2f}")
    print(f"Median: {split['total_demand'].median():.2f}")
    print(f"Min: {split['total_demand'].min():.2f}")
    print(f"Max: {split['total_demand'].max():.2f}")