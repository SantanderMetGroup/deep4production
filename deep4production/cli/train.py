## Load libraries
import os
import sys
import yaml
import json
import mlflow
from deep4production.utils.general import get_func_from_string
from deep4production.utils.log import setup_logging, get_logger
from deep4production.utils.paths import (
    resolve_id_dir,
    models_dir,
    aux_dir as aux_dir_for,
    predictions_dir,
)
from deep4production.utils.distributed import (
    init_distributed,
    cleanup_distributed,
    is_main_process,
    silence_non_main_ranks,
    barrier,
)


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
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # --- Multi-GPU / multi-node bootstrap ------------------------------------
    # Reads SLURM env vars (or torchrun fallback) and starts the NCCL process
    # group. Skipped automatically when num_nodes * gpus_per_node <= 1, so
    # single-GPU and CPU training go through the same path unchanged.
    hardware = config.get("hardware", None)
    dist_info = init_distributed(hardware)
    # Quiet non-zero ranks so SLURM logs only carry the rank-0 stream.
    silence_non_main_ranks()
    log.info("d4p train: starting")
    if dist_info["distributed"]:
        log.info(
            "Distributed training: rank=%d/%d local_rank=%d device=%s",
            dist_info["rank"],
            dist_info["world_size"],
            dist_info["local_rank"],
            dist_info["device"],
        )

    try:
        # --- Unpack config to get parameters ------------------------------------------
        data = config["data"]
        dataloader = config["dataloader"]
        model_info = config["model_info"]
        # output_dir + run_ID are mandatory: together they define the run
        # directory id_dir = output_dir/run_ID, which IS the model directory
        # (holds the recipes/scripts) and whose outputs/ subtree receives every
        # generated artifact. There is no default — opting out of the convention
        # is not allowed.
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
        overwrite = config.get("overwrite", False)
        graph = config.get("graph", None)
        Mlflow = config.get("Mlflow", None)
        tracker = config.get("tracker", None)

        # --- Monitoring backend: MLflow and d4p-tracker are mutually exclusive
        # — pick at most one. They both consume the validation-period prediction
        # machinery, so allowing both would double the work for no benefit.
        if Mlflow is not None and tracker is not None:
            log.error(
                "Both 'Mlflow' and 'tracker' are configured in the recipe; they "
                "are mutually exclusive. Keep only one."
            )
            sys.exit(1)

        # --- Create directories ----------------------------------
        # id_dir = output_dir/run_ID is the run directory; every generated
        # artifact lives under its outputs/ subtree (models/, aux_files/,
        # predictions/, tracker/) — see deep4production.utils.paths.
        id_dir = resolve_id_dir(output_dir, run_ID)
        model_dir = models_dir(id_dir)
        aux_dir = aux_dir_for(id_dir)
        pred_dir = predictions_dir(id_dir)
        # Only rank 0 creates the directory tree; other ranks wait at a barrier
        # so they see the dirs before reading/writing into them.
        if is_main_process():
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(aux_dir, exist_ok=True)
            os.makedirs(pred_dir, exist_ok=True)
        barrier()

        # --- Import training module ----------------------------------
        d4dt = config.get("d4p_trainer", None)
        if d4dt is None:
            d4p_trainer = get_func_from_string(
                "deep4production.core.trainers.trainer", "trainer"
            )
        else:
            d4p_trainer = get_func_from_string(d4dt["module"], d4dt["name"])
        d4dpy = config.get("d4p_pydataset", {})
        kwargs_trainer = config.get("d4p_trainer", {}).get("kwargs", {})

        # --- Start Mlflow and log config (rank 0 only) -----------------------
        # MLflow autologging, run starts and artifact uploads only happen on the
        # main process. The trainer wraps the active backend in a Monitor that
        # is a no-op on non-main ranks (see deep4production.utils.monitors), so
        # monitoring I/O stays on rank 0.
        if Mlflow is not None and is_main_process():
            ## Set tracking uri, i.e., MLFlow server
            tracking_uri = Mlflow["tracking_uri"]
            mlflow.set_tracking_uri(tracking_uri)
            ## Credentials
            usr = Mlflow.get("username", None)
            pwd = Mlflow.get("password", None)
            if usr is not None and pwd is not None:
                os.environ["MLFLOW_TRACKING_USERNAME"] = usr
                os.environ["MLFLOW_TRACKING_PASSWORD"] = pwd
            log.info(
                "Connected to MLflow tracking server: %s", mlflow.get_tracking_uri()
            )
            ## Set experiment within MLFlow
            experiment = Mlflow["experiment"]
            mlflow.set_experiment(experiment)
            log.info("MLflow experiment set: %s", experiment)
            ## Logs: system metrics
            mlflow.pytorch.autolog(disable=True)
            mlflow.enable_system_metrics_logging()
            ## Set run within experiment:
            run_name = Mlflow["run"]
            mlflow.start_run(run_name=run_name)
            ## Logs: yaml conf
            mlflow.log_params(
                {
                    "data": json.dumps(data),
                    "dataloader": json.dumps(dataloader),
                    "model_info": json.dumps(model_info),
                    "graph": json.dumps(graph),
                    "hardware": json.dumps(hardware),
                }
            )
            ## Artifact: yaml conf
            with open("config.yaml", "w") as f:
                yaml.dump(
                    {
                        "data": data,
                        "dataloader": dataloader,
                        "id_dir": id_dir,
                        "model_info": model_info,
                        "graph": graph,
                        "d4dpy": d4dpy,
                        "output_dir": output_dir,
                        "overwrite": overwrite,
                        "Mlflow": Mlflow,
                        "hardware": hardware,
                    },
                    f,
                    indent=2,
                )

            mlflow.log_artifact("config.yaml", artifact_path="config")
        # The trainer builds a no-op Monitor on non-main ranks regardless, but
        # we also withhold the backend configs here so only rank 0 ever drives
        # monitoring (MLflow run + tracker filesystem writes).
        trainer_mlflow = Mlflow if is_main_process() else None
        trainer_tracker = tracker if is_main_process() else None

        # --- Extract normalizer blocks from the recipe and pass them directly
        # to the trainer. The pydataset no longer applies normalization per-sample
        # on CPU; the trainer's GPU-side InputNormalizer modules handle it. The
        # recipe schema is unchanged — these are the same dicts that pydataset
        # used to consume.
        normalizer_info_x = data.get("predictors", {}).get("normalizer", None)
        normalizer_info_y = data.get("predictands", {}).get("normalizer", None)
        normalizer_info_f = (
            data.get("forcings", {}).get("normalizer", None)
            if data.get("forcings")
            else None
        )

        # --- Train ----------------------------------
        model_path = f"{model_dir}/{model_info["saving_params"]["model_save_name"]}.pt"
        if not os.path.exists(model_path) or overwrite:
            kwargs_trainer = {
                **kwargs_trainer,
                "data": data,
                "dataloader": dataloader,
                "id_dir": id_dir,
                "model_info": model_info,
                "graph": graph,
                "d4dpy": d4dpy,
                "Mlflow": trainer_mlflow,
                "tracker": trainer_tracker,
                "normalizer_info_x": normalizer_info_x,
                "normalizer_info_y": normalizer_info_y,
                "normalizer_info_f": normalizer_info_f,
                "hardware": hardware,
            }
            trainer = d4p_trainer(**kwargs_trainer)
            train_dataset, valid_dataset = trainer.get_pydatasets()
            train_dataloader, valid_dataloader = trainer.get_dataloaders(
                train_dataset, valid_dataset
            )
            trainer.train(train_dataloader, valid_dataloader)
        else:
            log.info(
                "Model %s already trained, available at: %s",
                model_info["saving_params"]["model_save_name"],
                model_path,
            )
    finally:
        # Always tear down the process group, including on early errors, so
        # SLURM tasks don't hang on a stale NCCL handle.
        cleanup_distributed()
