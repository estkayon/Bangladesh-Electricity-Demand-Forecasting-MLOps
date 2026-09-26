import os

import mlflow

from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from src.logger import get_logger


logger = get_logger("register_bridge_model")


MODEL_NAME = (
    "bangladesh-electricity-demand-bridge"
)

ALIAS_NAME = "champion"


def main():
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

    client = MlflowClient()

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    if not versions:
        raise ValueError(
            "No Bridge model versions found"
        )

    latest_version = max(
        versions,
        key=lambda item: int(
            item.version
        ),
    )

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=ALIAS_NAME,
        version=str(
            latest_version.version
        ),
    )

    print(
        "\nBRIDGE MODEL REGISTRY STATUS\n"
    )

    print(
        f"{MODEL_NAME} | "
        f"Version {latest_version.version} | "
        f"Alias: {ALIAS_NAME}"
    )

    logger.info(
        "Bridge model champion alias "
        "assigned successfully"
    )


if __name__ == "__main__":
    main()