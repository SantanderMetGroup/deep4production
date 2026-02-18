## Load libraries
import os
import sys
import yaml
import json
import mlflow
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
    print("👋 WELCOME TO D4D TRAIN!")
    data = config["data"]
    dataloader = config["dataloader"]
    model_info = config["model_info"]
    run_ID = config["run_ID"]
    output_dir = config.get("output_dir", "./")
    overwrite = config.get("overwrite", False)
    graph = config.get("graph", None)
    Mlflow = config.get("Mlflow", None)

    # --- Assign run ID ----------------------------------
    if run_ID is None:
      run_ID = ''.join(random.choices(string.ascii_letters + string.digits, k=5))

    # --- Create directories ----------------------------------
    id_dir = os.path.abspath(f"{output_dir}/{run_ID}")
    model_dir = f"{id_dir}/models/"
    os.makedirs(model_dir, exist_ok=True)
    metadata_dir = f"{id_dir}/metadata/"
    os.makedirs(metadata_dir, exist_ok=True)
    aux_dir = f"{id_dir}/aux_files/"
    os.makedirs(aux_dir, exist_ok=True)
    aux_prd = f"{id_dir}/predictions/"
    os.makedirs(aux_prd, exist_ok=True)
    
    # --- Import training module ----------------------------------
    d4dt = config.get("d4d_trainer", None)
    if d4dt is None: 
      d4d_trainer = get_func_from_string("deep4production.classes.d4d_trainer", "d4d_trainer")
    else:
      d4d_trainer = get_func_from_string(d4dt["module"], d4dt["name"])
    d4dpy = config.get("d4d_pydataset", {})

    # --- Start Mlflow and log config ----------------------------------
    if Mlflow is not None:
      ## Set tracking uri, i.e., MLFlow server
      tracking_uri = Mlflow["tracking_uri"]
      mlflow.set_tracking_uri(tracking_uri)
      ## Credentials
      usr = Mlflow.get("username", None)
      pwd = Mlflow.get("password", None)
      if usr is not None and pwd is not None:
          os.environ["MLFLOW_TRACKING_USERNAME"] = usr
          os.environ["MLFLOW_TRACKING_PASSWORD"] = pwd
      print(f"🌐 Connected to MLflow Tracking Server: {mlflow.get_tracking_uri()}")
      ## Set experiment within MLFlow
      experiment = Mlflow["experiment"]
      mlflow.set_experiment(experiment)
      print(f"🌐 Successfully set or created experiment '{experiment}'")
      ## Logs: system metrics
      mlflow.pytorch.autolog(disable=True)
      mlflow.enable_system_metrics_logging()
      ## Set run within experiment:
      run_name = Mlflow["run"]
      run = mlflow.start_run(run_name=run_name)
      ## Logs: yaml conf
      mlflow.log_params({
          "data": json.dumps(data),
          "dataloader": json.dumps(dataloader),
          "model_info": json.dumps(model_info),
          "graph": json.dumps(graph)
      })
      ## Artifact: yaml conf
      with open("config.yaml", "w") as f:
          yaml.dump({
              "data": data,
              "dataloader": dataloader,
              "id_dir": id_dir,
              "model_info": model_info,
              "graph": graph,
              "d4dpy": d4dpy,
              "output_dir": output_dir,
              "overwrite": overwrite,
              "Mlflow": Mlflow
          }, f, indent=2)
          artifact_path = f.name
      mlflow.log_artifact("config.yaml", artifact_path="config")

    # --- Train ----------------------------------
    model_path=f"{model_dir}/{model_info["saving_params"]["model_save_name"]}.pt"
    if not os.path.exists(model_path) or overwrite:
      trainer = d4d_trainer(data, dataloader, id_dir, model_info, graph, d4dpy, Mlflow)
      train_dataset, valid_dataset = trainer.get_pydatasets()
      train_dataloader, valid_dataloader = trainer.get_dataloaders(train_dataset, valid_dataset)
      trainer.train(train_dataloader, valid_dataloader) 
    else:
        print(f"🤖 {model_info["model_save_name"]} READY AT: {model_path}")
