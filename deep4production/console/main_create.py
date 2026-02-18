import os
import sys
import yaml
from deep4production.classes.d4d_dataset import d4d_dataset

def main():

    # --- Get config from YAML ------------------------------------------
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # --- Unpack config to get parameters ------------------------------------------
    date_init = config["date_init"]
    date_end = config["date_end"]
    freq = config["freq"]
    data = config["data"]
    output_path = config.get("output_path", "./")
    overwrite = config.get("overwrite", False)

    # --- Log info ------------------------------------------
    print(f"""
    -----------------------------------------------------------------------------------------------------------
    WELCOME TO D4D CREATE DATASET! 📈🤖📊

    Date Init: {date_init}
    Date End: {date_end}
    Temporal freq.: {freq}
    Dataset will be saved here: {output_path}
    Overwrite: {overwrite}
    -----------------------------------------------------------------------------------------------------------
    """)  

    # --- Create dirs to store the outputs ------------------------------------------
    os.makedirs(output_path, exist_ok=True)

    # --- Create Zarr ------------------------------------------
    if not os.path.exists(output_path) or overwrite:
      os.makedirs(output_path, exist_ok=True)
      # Training dataset
      d = d4d_dataset(date_init, date_end, freq, data) # Call Init
      d.to_disk(zarr_path = output_path) # Preprocess dataset and save it as a torch dataset for rapid loading.
      # Log info
      print("----------------------------------------------------")
      print("✅  🤞 🎯 Dataset (.zarr) created successfully! 🎯  🤞 ✅")
    else:
        # Log info
        print("----------------------------------------------------")
        print(f"Dataset (.zarr) already exists at: {output_path}")


    
    
