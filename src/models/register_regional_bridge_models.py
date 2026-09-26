import os

import mlflow

from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from src.logger import get_logger


logger = get_logger(
    "register_regional_bridge_models"
)


MODEL_NAMES = [
    "bangladesh-electricity-demand-barisal-bridge",
    "bangladesh-electricity-demand-chittagong-bridge",
    "bangladesh-electricity-demand-comilla-bridge",
    "bangladesh-electricity-demand-dhaka-bridge",
    "bangladesh-electricity-demand-khulna-bridge",
    "bangladesh-electricity-demand-mymensingh-bridge",
    "bangladesh-electricity-demand-rajshahi-bridge",
    "bangladesh-electricity-demand-rangpur-bridge",
    "bangladesh-electricity-demand-sylhet-bridge",
]

ALIAS_NAME = "champion"


def setup_mlflow():
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
        "MLflow tracking configured"
    )


def get_latest_version(
    client,
    model_name,
):
    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    if not versions:
        raise ValueError(
            f"No model versions found for "
            f"{model_name}"
        )

    return max(
        versions,
        key=lambda item: int(
            item.version
        ),
    )


def main():
    setup_mlflow()

    client = MlflowClient()

    results = []

    for model_name in MODEL_NAMES:
        latest_version = (
            get_latest_version(
                client,
                model_name,
            )
        )

        client.set_registered_model_alias(
            name=model_name,
            alias=ALIAS_NAME,
            version=str(
                latest_version.version
            ),
        )

        logger.info(
            f"Champion alias assigned: "
            f"{model_name} "
            f"v{latest_version.version}"
        )

        results.append(
            {
                "model_name":
                    model_name,

                "version":
                    latest_version.version,

                "alias":
                    ALIAS_NAME,
            }
        )

    print(
        "\nREGIONAL BRIDGE MODEL "
        "REGISTRY STATUS\n"
    )

    for result in results:
        print(
            f"{result['model_name']} | "
            f"Version {result['version']} | "
            f"Alias: {result['alias']}"
        )

    logger.info(
        "All regional Bridge model aliases "
        "configured successfully"
    )


if __name__ == "__main__":
    main()