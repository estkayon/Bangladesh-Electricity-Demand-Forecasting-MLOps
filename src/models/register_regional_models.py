import os

import mlflow

from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from src.logger import get_logger


logger = get_logger("register_regional_models")

MODEL_NAMES = [
    "bangladesh-electricity-demand-chittagong-ridge",
    "bangladesh-electricity-demand-comilla-ridge",
    "bangladesh-electricity-demand-dhaka-ridge",
    "bangladesh-electricity-demand-khulna-ridge",
    "bangladesh-electricity-demand-mymensingh-ridge",
    "bangladesh-electricity-demand-rangpur-ridge",
    "bangladesh-electricity-demand-sylhet-ridge",
]

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
        "MLflow tracking URI configured"
    )


def get_latest_version(
    client,
    model_name
):
    logger.info(
        f"Fetching latest version for: "
        f"{model_name}"
    )

    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    if not versions:
        raise ValueError(
            f"No registered versions found "
            f"for model: {model_name}"
        )

    latest_version = max(
        versions,
        key=lambda version: int(
            version.version
        )
    )

    return latest_version


def main():
    setup_mlflow()

    client = MlflowClient()

    results = []

    for model_name in MODEL_NAMES:
        latest_version = get_latest_version(
            client,
            model_name
        )

        logger.info(
            f"Assigning alias "
            f"'{ALIAS_NAME}' to "
            f"{model_name} "
            f"version "
            f"{latest_version.version}"
        )

        client.set_registered_model_alias(
            name=model_name,
            alias=ALIAS_NAME,
            version=str(
                latest_version.version
            ),
        )

        model_uri = (
            f"models:/{model_name}"
            f"@{ALIAS_NAME}"
        )

        results.append(
            {
                "model_name": model_name,
                "version":
                    latest_version.version,
                "alias": ALIAS_NAME,
                "model_uri": model_uri,
            }
        )

        logger.info(
            f"Alias assigned successfully: "
            f"{model_name}"
        )

    print(
        "\nREGIONAL MODEL REGISTRY STATUS\n"
    )

    for result in results:
        print(
            f"{result['model_name']} | "
            f"Version {result['version']} | "
            f"Alias: {result['alias']}"
        )

    logger.info(
        "All regional model aliases "
        "configured successfully"
    )


if __name__ == "__main__":
    main()