import os

import mlflow

from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from src.logger import get_logger


logger = get_logger("register_model")

MODEL_NAME = "bangladesh-electricity-demand-ridge"
ALIAS_NAME = "champion"


def setup_mlflow():
    logger.info(
        "Loading MLflow configuration"
    )

    load_dotenv()

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI"
    )

    if not tracking_uri:
        raise ValueError(
            "MLFLOW_TRACKING_URI not found"
        )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    logger.info(
        f"MLflow tracking URI configured"
    )


def get_latest_model_version(client):
    logger.info(
        f"Fetching latest version for: {MODEL_NAME}"
    )

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    if not versions:
        raise ValueError(
            f"No registered versions found "
            f"for model: {MODEL_NAME}"
        )

    latest_version = max(
        versions,
        key=lambda version: int(
            version.version
        )
    )

    logger.info(
        f"Latest registered version: "
        f"{latest_version.version}"
    )

    return latest_version


def assign_champion_alias(
    client,
    version
):
    logger.info(
        f"Assigning alias '{ALIAS_NAME}' "
        f"to version {version}"
    )

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=ALIAS_NAME,
        version=str(version),
    )

    logger.info(
        f"Alias '{ALIAS_NAME}' assigned "
        f"successfully"
    )


def main():
    setup_mlflow()

    client = MlflowClient()

    latest_version = (
        get_latest_model_version(
            client
        )
    )

    assign_champion_alias(
        client,
        latest_version.version
    )

    model_uri = (
        f"models:/{MODEL_NAME}"
        f"@{ALIAS_NAME}"
    )

    print(
        "\nMODEL REGISTRY STATUS"
    )

    print(
        f"Model name: {MODEL_NAME}"
    )

    print(
        f"Champion version: "
        f"{latest_version.version}"
    )

    print(
        f"Model URI: {model_uri}"
    )

    logger.info(
        "Model registry configuration completed"
    )


if __name__ == "__main__":
    main()