## Load libraries
import os
import sys
import yaml
from deep4production.utils.general import get_func_from_string

def main():
    # --- Check .sh call ------------------------------------------
    if len(sys.argv) != 2:
        print("Usage: d4d-train path/to/config.yaml")
        sys.exit(1)  # Exit with error code

    # --- Get config from YAML ------------------------------------------
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # --- Unpack config to get parameters ------------------------------------------
    print("👋 WELCOME TO D4D DOWNSCALE!")
    id_dir = config["id_dir"]
    input_data = config["input_data"]
    graph = config.get("graph", None)
    ensemble_size = config["ensemble_size"]
    model_file = config["model_file"]
    saving_info = config["saving_info"]

    # --- Import downscaler module ----------------------------------
    d4dp = config.get("d4d_downscaler", None)
    if d4dp is None: 
      d4d_downscaler = get_func_from_string("deep4production.classes.d4d_downscaler", "d4d_downscaler")
    else:
      d4d_downscaler = get_func_from_string(d4dp["module"], d4dp["name"])

    # --- Downscale ----------------------------------
    downscaler = d4d_downscaler(id_dir=id_dir, input_data=input_data, graph=graph, ensemble_size=ensemble_size, model_file=model_file, saving_info=saving_info)
    downscaler.downscale()
    
