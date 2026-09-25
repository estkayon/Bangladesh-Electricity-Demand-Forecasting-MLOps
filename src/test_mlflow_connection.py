import os

import mlflow
from dotenv import load_dotenv

from src.logger import get_logger


logger = get_logger("mlflow_connection")

load_dotenv()

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")

if not tracking_uri:
    raise ValueError("MLFLOW_TRACKING_URI not found in .env")

if not username:
    raise ValueError("MLFLOW_TRACKING_USERNAME not found in .env")

if not password:
    raise ValueError("MLFLOW_TRACKING_PASSWORD not found in .env")

logger.info("Environment variables loaded successfully")

mlflow.set_tracking_uri(tracking_uri)

logger.info("Testing MLflow connection")

experiment_name = "connection-test"

mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    mlflow.log_param("test_param", "connection_ok")
    mlflow.log_metric("test_metric", 1.0)

logger.info("MLflow connection test completed successfully")