import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.logger import get_logger


logger = get_logger("train")

DATA_PATH = "data/processed/model_features.csv"


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    mape = np.mean(
        np.abs((y_true - y_pred) / y_true)
    ) * 100

    return mae, rmse, mape


def main():
    logger.info("Loading feature dataset")

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    logger.info("Creating chronological train/validation/test splits")

    train = df[df["Date"] <= "2024-12-31"].copy()

    validation = df[
        (df["Date"] >= "2025-01-01")
        & (df["Date"] <= "2025-12-31")
    ].copy()

    test = df[
        df["Date"] >= "2026-01-01"
    ].copy()

    logger.info(f"Train rows: {len(train)}")
    logger.info(f"Validation rows: {len(validation)}")
    logger.info(f"Test rows: {len(test)}")

    logger.info("Running naive persistence baseline")

    # Tomorrow's demand = today's demand
    y_val = validation["next_day_total_demand"]
    val_pred = validation["total_demand"]

    val_mae, val_rmse, val_mape = calculate_metrics(
        y_val,
        val_pred
    )

    logger.info("Validation Baseline Results")
    logger.info(f"MAE: {val_mae:.2f} MW")
    logger.info(f"RMSE: {val_rmse:.2f} MW")
    logger.info(f"MAPE: {val_mape:.2f}%")

    y_test = test["next_day_total_demand"]
    test_pred = test["total_demand"]

    test_mae, test_rmse, test_mape = calculate_metrics(
        y_test,
        test_pred
    )

    logger.info("Test Baseline Results")
    logger.info(f"MAE: {test_mae:.2f} MW")
    logger.info(f"RMSE: {test_rmse:.2f} MW")
    logger.info(f"MAPE: {test_mape:.2f}%")


if __name__ == "__main__":
    main()