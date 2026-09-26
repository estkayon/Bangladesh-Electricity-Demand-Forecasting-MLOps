import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.logger import get_logger


logger = get_logger("error_analysis")

DATA_PATH = "data/processed/model_features.csv"

FEATURE_COLUMNS = [
    "total_demand",
    "total_load_shed",
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
    "trend_days",
    "lag_1_day",
    "lag_7_day",
    "lag_14_day",
    "rolling_7_day_mean",
    "rolling_14_day_mean",
    "rolling_30_day_mean",
    "load_shed_lag_1_day",
    "day_of_year",
    "is_holiday",
]

TARGET_COLUMN = "next_day_total_demand"


def calculate_errors(y_true, y_pred):
    absolute_error = np.abs(y_true - y_pred)

    percentage_error = (
        absolute_error / y_true
    ) * 100

    return absolute_error, percentage_error


def main():
    logger.info("Loading feature dataset")

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    train = df[
        df["Date"] <= "2024-12-31"
    ].copy()

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

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    results = test[
        [
            "Date",
            "total_demand",
            "total_load_shed",
            TARGET_COLUMN,
        ]
    ].copy()

    # Baseline
    logger.info("Generating baseline predictions")

    results["baseline_pred"] = test["total_demand"].values

    # Ridge
    logger.info("Training Ridge model")

    ridge_model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "ridge",
                Ridge(alpha=0.01)
            ),
        ]
    )

    ridge_model.fit(X_train, y_train)

    results["ridge_pred"] = ridge_model.predict(X_test)

    # Random Forest
    logger.info("Training Random Forest model")

    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    rf_model.fit(X_train, y_train)

    results["rf_pred"] = rf_model.predict(X_test)

    # XGBoost
    logger.info("Training XGBoost model")

    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )

    xgb_model.fit(X_train, y_train)

    results["xgb_pred"] = xgb_model.predict(X_test)

    # Error columns
    logger.info("Calculating model errors")

    model_columns = {
        "baseline": "baseline_pred",
        "ridge": "ridge_pred",
        "random_forest": "rf_pred",
        "xgboost": "xgb_pred",
    }

    for model_name, prediction_column in model_columns.items():
        abs_error, pct_error = calculate_errors(
            results[TARGET_COLUMN],
            results[prediction_column],
        )

        results[f"{model_name}_abs_error"] = abs_error
        results[f"{model_name}_pct_error"] = pct_error

    # Summary
    print("\nMODEL ERROR SUMMARY")

    for model_name in model_columns:
        mae = results[
            f"{model_name}_abs_error"
        ].mean()

        mape = results[
            f"{model_name}_pct_error"
        ].mean()

        print(
            f"{model_name:15} "
            f"MAE: {mae:.2f} MW | "
            f"MAPE: {mape:.2f}%"
        )

    # Biggest baseline error dates
    print("\nTOP 10 BASELINE ERROR DATES")

    print(
        results.nlargest(
            10,
            "baseline_abs_error"
        )[
            [
                "Date",
                "total_demand",
                TARGET_COLUMN,
                "baseline_pred",
                "baseline_abs_error",
                "baseline_pct_error",
            ]
        ].to_string(index=False)
    )

    # Biggest Ridge error dates
    print("\nTOP 10 RIDGE ERROR DATES")

    print(
        results.nlargest(
            10,
            "ridge_abs_error"
        )[
            [
                "Date",
                "total_demand",
                TARGET_COLUMN,
                "ridge_pred",
                "ridge_abs_error",
                "ridge_pct_error",
            ]
        ].to_string(index=False)
    )

    # Feature importance
    print("\nRANDOM FOREST FEATURE IMPORTANCE")

    rf_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": rf_model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False
    )

    print(rf_importance.to_string(index=False))

    print("\nXGBOOST FEATURE IMPORTANCE")

    xgb_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": xgb_model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False
    )

    print(xgb_importance.to_string(index=False))

    output_path = "artifacts/error_analysis.csv"

    results.to_csv(
        output_path,
        index=False
    )

    logger.info(
        f"Error analysis saved to: {output_path}"
    )

    logger.info(
        "Error analysis completed successfully"
    )


if __name__ == "__main__":
    main()