import subprocess
import sys
import time

from src.logger import get_logger


logger = get_logger(
    "refresh_live_forecasts"
)


# ============================================================
# Pipeline Configuration
# ============================================================

PIPELINE_STEPS = [
    {
        "name": "Collect BPDB Data",
        "module": "src.data.collect_data",
    },
    {
        "name": "Preprocess BPDB Data",
        "module": "src.data.preprocess",
    },
    {
        "name": "Refresh Weather Data",
        "module": "src.data.collect_weather",
    },
    {
        "name": "Build Live Inference Features",
        "module": "src.features.build_inference_features",
    },
    {
        "name": "Generate National Bridge Forecast",
        "module": "src.models.generate_bridge_forecast",
    },
    {
        "name": "Generate Regional Bridge Forecast",
        "module": "src.models.generate_regional_bridge_forecast",
    },
]


# ============================================================
# Run One Step
# ============================================================

def run_step(
    step_number,
    total_steps,
    step_name,
    module_name,
):
    logger.info(
        "=" * 70
    )

    logger.info(
        f"STEP {step_number}/{total_steps}: "
        f"{step_name}"
    )

    logger.info(
        f"Running module: "
        f"{module_name}"
    )

    start_time = time.time()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            module_name,
        ],
        check=False,
    )

    elapsed_time = (
        time.time()
        - start_time
    )

    if result.returncode != 0:
        logger.error(
            f"FAILED: {step_name}"
        )

        logger.error(
            f"Module: {module_name}"
        )

        logger.error(
            f"Exit code: "
            f"{result.returncode}"
        )

        logger.error(
            f"Elapsed time: "
            f"{elapsed_time:.2f} seconds"
        )

        raise RuntimeError(
            f"Pipeline stopped because "
            f"'{step_name}' failed."
        )

    logger.info(
        f"COMPLETED: {step_name}"
    )

    logger.info(
        f"Elapsed time: "
        f"{elapsed_time:.2f} seconds"
    )


# ============================================================
# Pipeline Summary
# ============================================================

def print_pipeline_summary(
    total_elapsed_time,
):
    print(
        "\nLIVE FORECAST REFRESH SUMMARY\n"
    )

    print(
        "Status: SUCCESS"
    )

    print(
        f"Steps completed: "
        f"{len(PIPELINE_STEPS)}"
    )

    print(
        f"Total runtime: "
        f"{total_elapsed_time:.2f} seconds"
    )

    print(
        "\nPipeline:"
    )

    for index, step in enumerate(
        PIPELINE_STEPS,
        start=1,
    ):
        print(
            f"{index}. "
            f"{step['name']}"
        )

    print(
        "\nForecast outputs refreshed:"
    )

    print(
        "- National live inference features"
    )

    print(
        "- Regional live inference features"
    )

    print(
        "- National extended forecast"
    )

    print(
        "- Regional extended forecasts"
    )


# ============================================================
# Main
# ============================================================

def main():
    logger.info(
        "Starting live forecast refresh pipeline"
    )

    pipeline_start = time.time()

    total_steps = len(
        PIPELINE_STEPS
    )

    try:
        for step_number, step in enumerate(
            PIPELINE_STEPS,
            start=1,
        ):
            run_step(
                step_number=step_number,
                total_steps=total_steps,
                step_name=step["name"],
                module_name=step["module"],
            )

    except Exception as error:
        total_elapsed_time = (
            time.time()
            - pipeline_start
        )

        logger.exception(
            "Live forecast refresh "
            "pipeline failed"
        )

        print(
            "\nLIVE FORECAST REFRESH SUMMARY\n"
        )

        print(
            "Status: FAILED"
        )

        print(
            f"Reason: {error}"
        )

        print(
            f"Runtime before failure: "
            f"{total_elapsed_time:.2f} seconds"
        )

        sys.exit(1)

    total_elapsed_time = (
        time.time()
        - pipeline_start
    )

    print_pipeline_summary(
        total_elapsed_time
    )

    logger.info(
        "Live forecast refresh pipeline "
        "completed successfully"
    )


if __name__ == "__main__":
    main()