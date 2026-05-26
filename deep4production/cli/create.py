import os
import sys
import yaml
from deep4production.core.datasets.dataset import dataset
from deep4production.utils.log import setup_logging, get_logger


def main():
    """
    Main entry point for the D4P dataset creation console script.
    Loads configuration from YAML, initializes dataset, and saves processed
    data to disk.

    Reads ``sys.argv[1]`` as the config file path. Returns ``None``.
    """
    setup_logging(level=os.environ.get("D4P_LOG_LEVEL", "INFO"))
    log = get_logger("cli.create")

    # --- Get config from YAML ------------------------------------------
    config_file = sys.argv[1]
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # --- Unpack config -------------------------------------------------
    date_init = config["date_init"]
    date_end = config["date_end"]
    freq = config["freq"]
    data = config["data"]
    output_path = config.get("output_path", "./")
    overwrite = config.get("overwrite", False)

    log.info("d4p create: %s → %s @ %s", date_init, date_end, freq)
    log.info("Output path: %s (overwrite=%s)", output_path, overwrite)

    # --- Create output dir --------------------------------------------
    os.makedirs(output_path, exist_ok=True)

    # --- Build & write zarr -------------------------------------------
    if not os.path.exists(output_path) or overwrite:
        os.makedirs(output_path, exist_ok=True)
        d = dataset(date_init, date_end, freq, data)
        d.to_disk(zarr_path=output_path)
        log.info("Dataset (.zarr) created successfully.")
    else:
        log.info("Dataset (.zarr) already exists at: %s", output_path)
