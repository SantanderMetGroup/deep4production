## Load libraries
import os
import sys
import yaml
import json
import random
import string
import mlflow
from deep4production.utils.general import get_func_from_string
from deep4production.utils.log import setup_logging, get_logger

def main():
    """
    Main entry point for the D4P training console script.
    Purpose: Loads configuration from YAML, initializes trainer, sets up MLflow, and runs the training process.
    Parameters:
        None (reads sys.argv for config file path)
    Returns:
        None
    """
    
    setup_logging(level=os.environ.get("D4P_LOG_LEVEL", "INFO"))
    log = get_logger("cli.train")

    # --- Check .sh call ------------------------------------------
    if len(sys.argv) != 2:
        log.error("Usage: d4p-train path/to/config.yaml")
        sys.exit(1)

    # --- Get config from YAML ------------------------------------------
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # --- Unpack config to get parameters ------------------------------------------
    log.info("d4p train: starting")
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
    aux_dir = f"{id_dir}/aux_files/"
    os.makedirs(aux_dir, exist_ok=True)
    pred_dir = f"{id_dir}/predictions/"
    os.makedirs(pred_dir, exist_ok=True)
    
    # --- Import training module ----------------------------------
    d4dt = config.get("d4p_trainer", None)
    if d4dt is None: 
      d4p_trainer = get_func_from_string("deep4production.core.trainers.trainer", "trainer")
    else:
      d4p_trainer = get_func_from_string(d4dt["module"], d4dt["name"])
    d4dpy = config.get("d4p_pydataset", {})
    kwargs_trainer = config.get("d4p_trainer", {}).get("kwargs", {})

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
      log.info("Connected to MLflow tracking server: %s", mlflow.get_tracking_uri())
      ## Set experiment within MLFlow
      experiment = Mlflow["experiment"]
      mlflow.set_experiment(experiment)
      log.info("MLflow experiment set: %s", experiment)
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

    # --- Extract normalizer blocks from the recipe and pass them directly
    # to the trainer. The pydataset no longer applies normalization per-sample
    # on CPU; the trainer's GPU-side InputNormalizer modules handle it. The
    # recipe schema is unchanged — these are the same dicts that pydataset
    # used to consume.
    normalizer_info_x = data.get("predictors",  {}).get("normalizer", None)
    normalizer_info_y = data.get("predictands", {}).get("normalizer", None)
    normalizer_info_f = data.get("forcings",    {}).get("normalizer", None) if data.get("forcings") else None

    # --- Train ----------------------------------
    model_path=f"{model_dir}/{model_info["saving_params"]["model_save_name"]}.pt"
    if not os.path.exists(model_path) or overwrite:
      kwargs_trainer = {**kwargs_trainer, "data": data, "dataloader": dataloader, "id_dir": id_dir, "model_info": model_info, "graph": graph, "d4dpy": d4dpy, "Mlflow": Mlflow,
                        "normalizer_info_x": normalizer_info_x, "normalizer_info_y": normalizer_info_y, "normalizer_info_f": normalizer_info_f}
      trainer = d4p_trainer(**kwargs_trainer)
      train_dataset, valid_dataset = trainer.get_pydatasets()
      train_dataloader, valid_dataloader = trainer.get_dataloaders(train_dataset, valid_dataset)
      trainer.train(train_dataloader, valid_dataloader) 
    else:
        log.info("Model %s already trained, available at: %s",
                 model_info["saving_params"]["model_save_name"], model_path)
