## Load libraries
import os
import sys
import yaml
from deep4production.utils.general import get_func_from_string

def main():
    """
    Main entry point for the D4P downscaling console script.
    Purpose: Loads configuration from YAML, initializes downscaler, and runs the downscaling process.
    Parameters:
        None (reads sys.argv for config file path)
    Returns:
        None
    """
    # --- Check .sh call ------------------------------------------
    if len(sys.argv) != 2:
        print("Usage: d4p-train path/to/config.yaml")
        sys.exit(1)  # Exit with error code

    # --- Get config from YAML ------------------------------------------
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # --- Unpack config to get parameters ------------------------------------------
    print("👋 WELCOME TO D4P DOWNSCALE!")
    id_dir = config["id_dir"]
    input_data = config["input_data"]
    graph = config.get("graph", None)
    ensemble_size = config["ensemble_size"]
    model_file = config["model_file"]
    saving_info = config["saving_info"]

    # --- Import downscaler module ----------------------------------
    d4p = config.get("d4p_downscaler", None)
    if d4p is None: 
      d4p_downscaler = get_func_from_string("deep4production.classes.d4p_downscaler", "d4p_downscaler")
    else:
      d4p_downscaler = get_func_from_string(d4p["module"], d4p["name"])

    # --- Downscale ----------------------------------
    downscaler = d4p_downscaler(id_dir=id_dir, input_data=input_data, graph=graph, ensemble_size=ensemble_size, model_file=model_file, saving_info=saving_info)
    downscaler.downscale()

