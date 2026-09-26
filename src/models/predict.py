import joblib
import pandas as pd

from src.logger import get_logger


logger = get_logger("predict")

MODEL_PATH = "artifacts/final_ridge_model.pkl"
FEATURE_DATA_PATH = "data/processed/model_features.csv"

FEATURE_COLUMNS = [
    "total_demand",
    "total_load_shed",
    "day_of_week",
    "month",
    "day_of_month",
    "day_of_year",
    "is_weekend",
    "is_holiday",
    "trend_days",
    "temperature_max_c",
    "temperature_min_c",
    "temperature_mean_c",
    "precipitation_mm",
    "rain_mm",
    "lag_1_day",
    "lag_7_day",
    "lag_14_day",
    "rolling_7_day_mean",
    "rolling_14_day_mean",
    "rolling_30_day_mean",
    "load_shed_lag_1_day",
]


def load_model():
    logger.info(
        f"Loading trained model from: {MODEL_PATH}"
    )

    model = joblib.load(
        MODEL_PATH
    )

    logger.info(
        "Model loaded successfully"
    )

    return model


def load_latest_features():
    logger.info(
        "Loading feature dataset"
    )

    df = pd.read_csv(
        FEATURE_DATA_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"]
    )

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "Feature dataset is empty"
        )

    latest_row = (
        df.iloc[-1]
        .copy()
    )

    logger.info(
        f"Latest observation date: "
        f"{latest_row['Date'].date()}"
    )

    logger.info(
        f"Forecast date: "
        f"{latest_row['forecast_date'].date()}"
    )

    return latest_row


def validate_features(latest_row):
    logger.info(
        "Validating prediction features"
    )

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in latest_row.index
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns: "
            f"{missing_features}"
        )

    feature_values = latest_row[
        FEATURE_COLUMNS
    ]

    if feature_values.isna().any():
        missing_values = (
            feature_values[
                feature_values.isna()
            ]
            .index
            .tolist()
        )

        raise ValueError(
            f"Missing feature values: "
            f"{missing_values}"
        )

    logger.info(
        "Prediction features validated successfully"
    )


def predict_next_day():
    model = load_model()

    latest_row = load_latest_features()

    validate_features(
        latest_row
    )

    X_latest = pd.DataFrame(
        [
            latest_row[
                FEATURE_COLUMNS
            ].to_dict()
        ]
    )

    logger.info(
        "Generating next-day demand prediction"
    )

    prediction = model.predict(
        X_latest
    )[0]

    observation_date = (
        latest_row["Date"].date()
    )

    forecast_date = (
        latest_row["forecast_date"].date()
    )

    current_demand = (
        latest_row["total_demand"]
    )

    logger.info(
        "Prediction completed successfully"
    )

    print(
        "\nNEXT-DAY ELECTRICITY DEMAND FORECAST"
    )

    print(
        f"Latest observation date: "
        f"{observation_date}"
    )

    print(
        f"Forecast date: "
        f"{forecast_date}"
    )

    print(
        f"Current national demand: "
        f"{current_demand:.2f} MW"
    )

    print(
        f"Predicted next-day demand: "
        f"{prediction:.2f} MW"
    )

    return {
        "observation_date": str(
            observation_date
        ),
        "forecast_date": str(
            forecast_date
        ),
        "current_demand_mw": float(
            current_demand
        ),
        "predicted_demand_mw": float(
            prediction
        ),
    }


if __name__ == "__main__":
    predict_next_day()