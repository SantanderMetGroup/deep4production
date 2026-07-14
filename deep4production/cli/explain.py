## Load libraries
import os
import sys
import yaml
from deep4production.utils.general import get_func_from_string
from deep4production.utils.log import setup_logging, get_logger
from deep4production.utils.paths import resolve_id_dir


def main():
    """
    Main entry point for the D4P explain (input-attribution) console script.
    Purpose: Loads configuration from YAML, initializes an explainer, and runs
        gradient-based input attribution, writing maps to id_dir/outputs/xai/.
    Parameters:
        None (reads sys.argv for config file path)
    Returns:
        None
    """
    setup_logging(level=os.environ.get("D4P_LOG_LEVEL", "INFO"))
    log = get_logger("cli.explain")

    # --- Check .sh call ------------------------------------------
    if len(sys.argv) != 2:
        log.error("Usage: d4p-explain path/to/config.yaml")
        sys.exit(1)

    # --- Get config from YAML ------------------------------------------
    config_file = sys.argv[1]
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # --- Unpack config to get parameters ------------------------------------------
    log.info("d4p explain: starting")
    # Same run directory as training/inference: id_dir = output_dir/run_ID. The
    # checkpoint is read from id_dir/outputs/models/ and attribution maps are
    # written to id_dir/outputs/xai/. Both keys are required (no id_dir key).
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
    ensemble_size = config.get("ensemble_size", 1)
    model_file = config["model_file"]
    saving_info = config["saving_info"]
    forcing_data = config.get("forcing_data", None)

    # --- Import explainer module ----------------------------------
    d4p = config.get("d4p_explainer", None)
    if d4p is None:
        explainer_cls = get_func_from_string(
            "deep4production.core.explainers.explainer", "Explainer"
        )
        kwargs_explainer = {}
    else:
        explainer_cls = get_func_from_string(d4p["module"], d4p["name"])
        kwargs_explainer = d4p.get("kwargs", {}) or {}

    # --- Attribution params (method, target_var, reduction, target_region, ...) ---
    explain_params = config.get("explain_params", {}) or {}

    # --- Explain ----------------------------------
    explainer = explainer_cls(
        id_dir=id_dir,
        input_data=input_data,
        graph=graph,
        ensemble_size=ensemble_size,
        model_file=model_file,
        saving_info=saving_info,
        forcing_data=forcing_data,
        **kwargs_explainer,
    )
    explainer.explain(**explain_params)
