## Load libraries
import os
import sys
import yaml
from deep4production.utils.general import get_func_from_string
from deep4production.utils.log import setup_logging, get_logger
from deep4production.utils.paths import resolve_id_dir


def main():
    """
    Main entry point for the D4P downscaling console script.
    Purpose: Loads configuration from YAML, initializes downscaler, and runs the downscaling process.
    Parameters:
        None (reads sys.argv for config file path)
    Returns:
        None
    """
    setup_logging(level=os.environ.get("D4P_LOG_LEVEL", "INFO"))
    log = get_logger("cli.downscale")

    # --- Check .sh call ------------------------------------------
    if len(sys.argv) != 2:
        log.error("Usage: d4p-downscale path/to/config.yaml")
        sys.exit(1)

    # --- Get config from YAML ------------------------------------------
    config_file = sys.argv[1]
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # --- Unpack config to get parameters ------------------------------------------
    log.info("d4p downscale: starting")
    # Inference shares the training convention: id_dir = output_dir/run_ID is the
    # run directory, and model_file / saving_info.file resolve under its outputs/
    # subtree. Both keys are required — there is no standalone id_dir key anymore.
    run_ID = config.get("run_ID", None)
    output_dir = config.get("output_dir", None)
    if not output_dir or not run_ID:
        log.error(
            "Both 'output_dir' and 'run_ID' are required in the recipe "
            "(id_dir = output_dir/run_ID). Missing: %s",
            ", ".join(
                k
                for k, v in (("output_dir", output_dir), ("run_ID", run_ID))
                if not v
            ),
        )
        sys.exit(1)
    id_dir = resolve_id_dir(output_dir, run_ID)
    input_data = config["input_data"]
    graph = config.get("graph", None)
    ensemble_size = config["ensemble_size"]
    model_file = config["model_file"]
    saving_info = config["saving_info"]
    forcing_data = config.get("forcing_data", None)
    # Optional per-variable physical clamp applied in physical space at the end
    # of postprocessing (e.g. {hurs: [0, 100]}). None → no clamping.
    physical_bounds = config.get("physical_bounds", None)
    # Optional per-variable unit conversion applied to predictions at write-time
    # (e.g. {pr: {name: mm_day_to_flux}}). None → predictions kept as-is.
    unit_conversion = config.get("unit_conversion", None)

    # --- Import downscaler module ----------------------------------
    d4p = config.get("d4p_downscaler", None)
    if d4p is None:
        d4p_downscaler = get_func_from_string(
            "deep4production.core.downscalers.downscaler", "downscaler"
        )
        kwargs_downscaler = {}
    else:
        d4p_downscaler = get_func_from_string(d4p["module"], d4p["name"])
        kwargs_downscaler = d4p.get("kwargs", {}) or {}

    # --- Inference runtime params (batch_size, amp_dtype, compile, ...) ---
    # Forwarded to downscale(). Defaults preserve the original eager-mode,
    # batch-size-1 behaviour when the YAML has no inference_params block.
    inference_params = config.get("inference_params", {}) or {}

    # --- Downscale ----------------------------------
    downscaler = d4p_downscaler(
        id_dir=id_dir,
        input_data=input_data,
        graph=graph,
        ensemble_size=ensemble_size,
        model_file=model_file,
        saving_info=saving_info,
        forcing_data=forcing_data,
        physical_bounds=physical_bounds,
        unit_conversion=unit_conversion,
        **kwargs_downscaler,
    )
    downscaler.downscale(**inference_params)
