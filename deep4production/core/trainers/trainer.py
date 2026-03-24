## Load libraries
import os
import yaml
import math
import json
import time
import torch
import zarr
import importlib
import numpy as np
from functools import partial
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
## MLFlow
import mlflow
import mlflow.pytorch
# from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
# from mlflow.exceptions import MlflowException
## Deep4production
from deep4production.deep.train import update_params
from deep4production.deep.utils import EMA
from deep4production.deep.utils import save_model, resume_model, load_model
from deep4production.utils.general import get_func_from_string
from deep4production.utils.mlflow import *
##################################################################################################################################
class trainer:
    def __init__(self, data, dataloader, id_dir, model_info, graph=None, d4dpy={}, Mlflow=None):
        """
        Initializes the trainer class.
        
        Purpose:
            Sets up the trainer with data, dataloader, model info, graph, metadata, and MLflow tracking.
        
        Parameters:
            data (dict): Dataset configuration and paths.
            dataloader (dict): Dataloader parameters (batch size, shuffle, num_workers).
            id_dir (str): Directory for experiment outputs.
            model_info (dict): Model, loss, saving, and training parameters.
            graph (dict, optional): Graph configuration for GNN models.
            d4dpy (dict, optional): Custom pydataset configuration.
            Mlflow (dict, optional): MLflow tracking configuration.
        """
        print("🚀 STARTING D4P TRAINER")
        # --- SELF PARAMETERS ---------------------------------------
        self.data = data
        self.dataloader = dataloader
        self.model_info = model_info
        self.graph = graph
        self.saving_params = model_info["saving_params"]
        self.loss_params = model_info["loss_params"]
        self.model_params = model_info["model_params"]
        self.training_params = model_info["training_params"]
        self.d4dpy = d4dpy
        if d4dpy: # Is d4dpy dict not empty?
            self.pydataset = get_func_from_string(d4dpy["module"], d4dpy["name"])
            self.d4dpy = d4dpy["kwargs"]
        else:
            self.pydataset = get_func_from_string("deep4production.core.pydatasets.pydataset", "pydataset")

        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')

        self.id_dir = id_dir
        self.model_dir = f"{id_dir}/models/"
        self.aux_dir = f"{id_dir}/aux_files/"
        print("📦 SELF READY")

        # --- BUILD GRAPH ---------------------------------------
        self.kwargs_training = self.training_params.get("kwargs", {})
        self.graph_loc = {}
        if graph is not None:
            edge_index = get_func_from_string(module_string=graph["module"],func_string=graph["name"], kwargs=graph.get("kwargs", None))
            self.graph_loc["path"] = "edge_index.pt"
            torch.save(edge_index, f"{self.aux_dir}/{self.graph_loc["path"]}")
            print(f"📦 GRAPH READY: function {graph['name']} from {graph['module']}")
        else:
            self.graph_loc = None
        
        # --- LOSS FUNCTION ---------------------------------------
        self.loss_function = get_func_from_string(module_string=self.loss_params["module"],func_string=self.loss_params["name"], kwargs=self.loss_params.get("kwargs", None))
        print(f"📦 LOSS READY: {self.loss_params['name']} from {self.loss_params['module']}")

        # --- MODEL ---------------------------------------
        self.model_save_name = model_info["saving_params"]["model_save_name"]
        self.model = get_func_from_string(module_string=self.model_params["module"], func_string=self.model_params["name"], kwargs=self.model_params.get("kwargs", None))
        self.model_path = f"{self.model_dir}/{self.model_save_name}.pt"
        print(f"📦 MODEL READY: {self.model_params['name']} from {self.model_params['module']}")

        # --- CREATE AND SAVE METADATA ---------------------------------------
        self.metadata_dict = self.build_metadata()

        # --- Mlflow ---------------------------------------
        self.Mlflow = Mlflow
        if self.Mlflow is not None:
            ## Mlflow dirs
            # print(run.info.experiment_id)
            # print(run.info.artifact_uri)
            # print(run.info.run_id)
            ## Tags:
            tags = Mlflow.get("tags", {})
            for key, value in tags.items():
                if value is not None:
                    mlflow.set_tag(key, value)
            ## Mlflow diagnostics and saving info
            self.Mlflow_diagnostics = Mlflow.get("diagnostics", None)
            self.Mlflow_compute_diagnostics_every_n_epochs = Mlflow.get("compute_diagnostics_every_n_epochs", None)
            self.Mlflow_save_checkpoint_every_n_epochs = Mlflow.get("save_checkpoint_every_n_epochs", None)
            if self.Mlflow_diagnostics is not None:
                ## Get d4p_downscaler function
                d4p_name = Mlflow.get("func_name", "downscaler")
                d4p_module = Mlflow.get("func_module", "deep4production.core.downscalers.downscaler")
                self.d4p_func = get_func_from_string(module_string=d4p_module, func_string=d4p_name)
                self.input_data = {"paths": data["predictors"]["paths"], "years": data["validation_period"], "load_in_memory": data["load_in_memory"]}
                if data.get("forcings", None) is not None:
                    self.forcing_data = {"paths": data["predictands"]["paths"], "years": data["validation_period"], "load_in_memory": data["load_in_memory"]}
                else:
                    self.forcing_data = None

                
    # -------------------------------------------------------------------------
    def build_metadata(self):
        """
        Builds and returns a metadata dictionary containing model and loss parameters.
        
        Purpose:
            Collects and organizes model and loss configuration for tracking and reproducibility.
        
        Parameters:
            None (uses self attributes)
        Returns:
            dict: Metadata dictionary.
        """
        # --- INIT METADATA DICTIONARY ---
        metadata_dict = {}
        metadata_dict["id_dir"] = self.id_dir
        ### Loss parameters
        metadata_dict["loss_params"] = {}
        metadata_dict["loss_params"] = {k: v for k, v in self.loss_params.items() if k not in ["name", "module"]}
        metadata_dict["loss_params"]["name"] = self.loss_params["name"]
        metadata_dict["loss_params"]["module"] = self.loss_params["module"]
        ### Model parameters
        metadata_dict["model_params"] = {}
        metadata_dict["model_params"] = {k: v for k, v in self.model_params.items() if k not in ["name", "module"]}
        metadata_dict["model_params"]["name"] = self.model_params["name"]
        metadata_dict["model_params"]["module"] = self.model_params["module"]
        # --- RETURN ---
        return metadata_dict

    # -------------------------------------------------------------------------
    def cont_metadata(self, pydataset):
        """
        Updates metadata dictionary with additional information from the pydataset.
        
        Purpose:
            Adds variables, lagged info, spatial info, normalizer and operator parameters, and forcings to metadata.
        
        Parameters:
            pydataset: Dataset object with methods to extract relevant info.
        Returns:
            dict: Updated metadata dictionary.
        """
        ### Variables
        self.metadata_dict["vars_x"], self.metadata_dict["vars_y"] = pydataset.get_vars()
        ### Lagged info
        self.metadata_dict["num_lagged_x"], self.metadata_dict["num_lagged_y"] = pydataset.get_lagged_info()
        ### Spatial info
        self.metadata_dict["lats_y"], self.metadata_dict["lons_y"] = pydataset.get_coords()
        self.metadata_dict["transform_to_2D_x"], self.metadata_dict["transform_to_2D_y"] = pydataset.get_transform2D()
        self.metadata_dict["H_x"], self.metadata_dict["W_x"], self.metadata_dict["H_y"], self.metadata_dict["W_y"], = pydataset.get_spatial_dims()
        self.metadata_dict["G_x"], self.metadata_dict["G_y"] = pydataset.get_num_gridpoints()
        ### Normalizer parameters (cont.)
        if self.data["predictors"].get("normalizer", None) is not None:
            self.metadata_dict["normalizer_x"] = pydataset.get_normalizer_info(predictands=False)
        if self.data["predictands"].get("normalizer", None) is not None:
            self.metadata_dict["normalizer_y"] = pydataset.get_normalizer_info(predictands=True)
        ### Operator parameters (cont.)
        if self.data["predictors"].get("operator", None) is not None:
            self.metadata_dict["operator_x"] = pydataset.get_operator_info(predictands=False)
        if self.data["predictands"].get("operator", None) is not None:
            self.metadata_dict["operator_y"] = pydataset.get_operator_info(predictands=True)
        ### Forcings
        self.metadata_dict["vars_f"], self.metadata_dict["idx_vars_f"], self.metadata_dict["normalizer_f"], self.metadata_dict["operator_f"] = pydataset.get_forcings_info()
        ### Return
        return self.metadata_dict

    # -------------------------------------------------------------------------
    def get_pydatasets(self):
        """
        Creates training and validation pydataset objects, updates metadata, and prepares for MLflow diagnostics.
        
        Purpose:
            Instantiates pydataset objects for training and validation, updates metadata, and prepares MLflow targets.
        
        Parameters:
            None (uses self attributes)
        Returns:
            tuple: (train_dataset, valid_dataset)
        """
        ## Create pydatasets
        kwargs_pydataset = {"predictors": self.data["predictors"], "predictands": self.data["predictands"], "forcings": self.data.get("forcings", {}), "load_in_memory": self.data.get("load_in_memory", True)}
        kwargs_pydataset.update(**self.d4dpy)
        train_dataset = self.pydataset(temporal_period = self.data["training_period"], **kwargs_pydataset)
        valid_dataset = None
        if self.data.get("validation_period", None) is not None:
            valid_dataset = self.pydataset(temporal_period = self.data["validation_period"], **kwargs_pydataset)
            if self.Mlflow is not None:
                if self.Mlflow_diagnostics is not None:
                    self.tgt_mlflow = valid_dataset.get_target_samples()
        ### Update metadata and save it with the new information
        self.metadata_dict = self.cont_metadata(train_dataset) 
        # self.save_metadata(self.metadata_path)
        print("📦 PYDATASETS READY")
        return train_dataset, valid_dataset

    # -------------------------------------------------------------------------
    def get_dataloaders(self, train_dataset, valid_dataset):
        """
        Creates PyTorch DataLoader objects for training and validation datasets.
        
        Purpose:
            Sets up DataLoader objects using parameters from YAML config for efficient batch processing.
        
        Parameters:
            train_dataset: Training dataset object.
            valid_dataset: Validation dataset object (optional).
        Returns:
            tuple: (train_dataloader, valid_dataloader)
        """
        ## Some parameters
        num_workers = self.dataloader.get("num_workers", 0)
        if self.dataloader.get("num_workers", None) is None:
            print("⚠️ WARNING: Number of workers not specified in YAML. Using num_workers=0")
        shuffle = self.dataloader.get("shuffle", False)
        if self.dataloader.get("shuffle", None) is None:
            print("⚠️ WARNING: Shuffle not specified in YAML. Using shuffle=False")
        batch_size = self.dataloader.get("batch_size", 1)
        if self.dataloader.get("batch_size", None) is None:
            print("⚠️ WARNING: Batch size not specified in YAML. Using batch_size=1")
        kwargs_dataloader = {"batch_size": batch_size, "shuffle": shuffle, "num_workers": num_workers}
        ## Create DataLoaders
        if self.graph is not None:
            DL = PyGDataLoader
        else:
            DL = TorchDataLoader
        train_dataloader = DL(train_dataset, **kwargs_dataloader)
        valid_dataloader = None
        if valid_dataset is not None:
            valid_dataloader = DL(valid_dataset, **kwargs_dataloader)
        print("📦 DATALOADERS READY")
        return train_dataloader, valid_dataloader

    # -------------------------------------------------------------------------
    def get_num_trainable_parameters(self):
        """
        Returns the total number of trainable parameters in the model.
        
        Purpose:
            Useful for model size reporting and debugging.
        
        Parameters:
            None (uses self.model)
        Returns:
            int: Number of trainable parameters.
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    # -------------------------------------------------------------------------
    def model_backprop(self, model, data, optimizer, loss_function, device, is_this_training=True, **kwargs):
        """
        Performs a single forward and backward pass for a batch, computes loss, and optionally backpropagates.
        
        Purpose:
            Handles the core training step for one batch, including loss computation and gradient update.
        
        Parameters:
            model: PyTorch model.
            data: Tuple of input, target, and optional forcing arrays.
            optimizer: PyTorch optimizer.
            loss_function: Loss function callable.
            device: Device string ('cpu' or 'cuda').
            is_this_training (bool): Whether to perform backpropagation.
            **kwargs: Additional arguments.
        Returns:
            float: Loss value for the batch.
        """
        # --- Get arrays as defined in the pydataset class. ---
        x, y, f = data
        x = x.to(device)
        y = y.to(device)

        if f[0] != "N/A":
            f = f.to(device)
        else:
            B, Cy, *spatial = y.shape
            f = torch.zeros(B, Cy, *spatial, device=device)

        # --- Feed features to the denoiser deep learning model ---
        prediction = model(x, f)

        # --- Compute loss ---
        optimizer.zero_grad()
        loss = loss_function(target=y, output=prediction)

        # --- Backpropagation ---
        if is_this_training:
            loss.backward()

        # --- Return ---
        return loss.item() 

    # -------------------------------------------------------------------------
    def training_loop(
        self,
        training_params: dict,
        saving_params: dict,
        model: torch.nn.Module,
        loss_function,
        device,
        train_data: torch.utils.data.DataLoader,
        valid_data: torch.utils.data.DataLoader = None,
        ema_decay: float = None,
        metadata: dict = None,
        kwargs: dict = {},
    ) -> dict:
        """
        Runs the main training loop, handles optimizer, scheduler, early stopping, model saving, MLflow logging, and diagnostics.
        
        Purpose:
            Orchestrates the full training process, including validation, early stopping, model saving, and MLflow integration.
        
        Parameters:
            training_params (dict): Training configuration (epochs, optimizer, scheduler, etc).
            saving_params (dict): Model saving configuration.
            model (torch.nn.Module): Model to train.
            loss_function: Loss function callable.
            device: Device string ('cpu' or 'cuda').
            train_data: Training DataLoader.
            valid_data: Validation DataLoader (optional).
            ema_decay (float, optional): EMA decay rate.
            metadata (dict, optional): Metadata dictionary.
            kwargs (dict, optional): Additional arguments.
        Returns:
            tuple: (train_losses, valid_losses)
        """
        # --- Get some training parameters ------------------------------------------------
        num_epochs = training_params["num_epochs"]
        patience_early_stopping = training_params.get("patience_early_stopping", None)

        # --- Model to device ------------------------------------------------ 
        model = model.to(device)
        model_size = sum(p.numel() for p in model.parameters())
        model_mb = model_size * 4 / (1024**2)     # float32 = 4 bytes
        print(f" --> Model parameters: {model_size:,} ({model_mb:.2f} MB)")

        # --- Early stopping setup ------------------------------------------------
        best_val_loss = math.inf
        early_stopping_counter = 0
        use_early_stopping = patience_early_stopping is not None

        # --- Loss tracking --------------------------------------------------------
        train_losses = []
        valid_losses = []

        # --- Optimizer ------------------------------------------
        optimizer_params = training_params.get("optimizer_params", {})
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
        current_lr = optimizer_params["lr"]
        global_step = 0 # Number of samples processed so far during training
        epoch_ref = 0 # Relevant for saving model every n epochs
        step_ref = 0 # Relevant for saving model every n steps

        # --- Learning rate scheduler ------------------------------------------
        scheduler = None
        scheduler_params = training_params.get("scheduler_params", None)
        if scheduler_params is not None:
            # Get scheduler function selected in YAML from "torch.optim.lr_scheduler"
            scheduler_type = scheduler_params["type"]
            scheduler_func = get_func_from_string(module_string="torch.optim.lr_scheduler", func_string=scheduler_type)
            scheduler_kwargs = scheduler_params.get("kwargs", None)
            # Handle LambdaLR separately (needs a callable)
            if scheduler_type == "LambdaLR":
                lambda_name = scheduler_params.get("lr_lambda", None)
                if lambda_name is None:
                    raise ValueError("LambdaLR requires 'lr_lambda' parameter in config YAML")
                lr_lambda_func = get_func_from_string(module_string="deep4production.deep.schedulers", func_string=lambda_name)
                lr_lambda = partial(lr_lambda_func, **scheduler_kwargs) # Use functools.partial to freeze parameters
                # Instantiate scheduler properly
                scheduler = scheduler_func(optimizer, lr_lambda=lr_lambda)
            else: # All other schedulers
                scheduler = scheduler_func(optimizer, **scheduler_kwargs)
            print(f"📦 Loaded scheduler: {scheduler_type}")

        # --- Resume training from a pretrained checkpoint? ------------------------------------------
        epoch_init=0
        if saving_params.get("resume_checkpoint", None) is not None:
            path_checkpoint = f"{self.model_dir}/{saving_params["resume_checkpoint"]}"
            if os.path.exists(path_checkpoint):
                print(f"🚀 Resuming training from checkpoint: {path_checkpoint}")
                checkpoint = resume_model(path=path_checkpoint, model=model, optimizer=optimizer, scheduler=scheduler, device=device)
                epoch_init = epoch_ref = epoch = checkpoint['epoch']
                step_ref = global_step = checkpoint['global_step']
                train_losses = checkpoint.get('train_losses', [])
                valid_losses = checkpoint.get('valid_losses', [])
                best_val_loss = np.min(valid_losses)
                epoch_best_val_loss = np.where(valid_losses == best_val_loss)[0][0]
                early_stopping_counter = epoch - epoch_best_val_loss
                print("🚀 Resume training:")
                print(f"    checkpoint: {path_checkpoint}")
                print(f"    epoch: {epoch}")
                print(f"    global_step: {global_step}")
            else:
                print(f"⚠️ WARNING: Checkpoint specified for resuming training not found at {path_checkpoint}. Starting training from scratch.")
            
        # --- Ensemble Model Averaging (EMA) parameters ------------------------------------------
        ema = None
        if ema_decay is not None:
            ema = EMA(model, decay=ema_decay, device=device)

        # --- Mlflow counter ---
        epoch_ref_mlflow = 0
        epoch_ref_mlflow_diagnostic = 0

        # --- Loop over epochs ------------------------------------------
        print(f"🚀 Starting training for {num_epochs} epochs on {device.upper()}") 
        for epoch in range(epoch_init, num_epochs + 1):
            epoch_start = time.time()

            # -----------------------------------------------------------------------------------------
            # --- Training phase: Loop over batches ---------------------------------------------------
            num_batches = len(train_data)
            model.train()
            train_loss_epoch = 0
            for batch_data in train_data:
                train_loss_epoch += self.model_backprop(
                    model=model,
                    data=batch_data,
                    optimizer=optimizer,
                    loss_function=loss_function,
                    device=device,
                    is_this_training=True,
                    **kwargs
                )
                # --- Scheduler: Update learning rate, and optimizer and loss ---
                current_lr = update_params(optimizer=optimizer, lr=current_lr, scheduler=scheduler)
                global_step += 1

            # --- Store training loss ---
            train_loss = train_loss_epoch / num_batches
            train_losses.append(train_loss)
            if self.Mlflow is not None:
                mlflow.log_metric("train_loss_epoch", train_loss, step=int(epoch))
            
            # --- Update EMA? ---
            if ema is not None:
                ema.update()

            # -----------------------------------------------------------------------------------------
            # --- Validation phase: Loop over batches -------------------------------------------------
            val_loss = None
            if valid_data is not None:
                model.eval()
                with torch.no_grad():
                    val_loss_epoch = 0
                    for batch_data in valid_data:
                        num_batches = len(valid_data)
                        val_loss_epoch += self.model_backprop(
                            model=model,
                            data=batch_data,
                            optimizer=optimizer,
                            loss_function=loss_function,
                            device=device,
                            is_this_training=False,
                            **kwargs
                        )
                    val_loss = val_loss_epoch / num_batches
                    if self.Mlflow is not None:
                        mlflow.log_metric("val_loss_epoch", val_loss, step=int(epoch))
                valid_losses.append(val_loss)

            # --- Compute epoch time -----------------------------------------------
            epoch_time = np.round(time.time() - epoch_start, 2)

            # --- Build log message -------------------------------------------------
            timestamp = time.strftime("[%H:%M:%S]")
            log_msg = (
                f"{timestamp} Epoch {epoch:04d} | Step {global_step:10d} | Time: {epoch_time:5.2f}s "
                f"| LR: {current_lr:.2e} | Train Loss: {train_loss:.5f}"
            )
            if val_loss is not None:
                log_msg += f" | Val Loss: {val_loss:.5f}"

            # --- Early stopping ----------------------------------
            save_model_or_not = False
            if use_early_stopping and val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                    save_model_or_not = True
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience_early_stopping:
                      print(f"{timestamp} 🛑 Early stopping triggered after {epoch} epochs.")
                      break

            # --- Save model (general info & checks) ----------------------------------
            kwargs_save = {'epoch': epoch,
                    'global_step': global_step,
                    'train_losses': train_losses,      
                    'valid_losses': valid_losses if valid_data else None,      
                    'model': model,
                    'optimizer': optimizer,  
                    'scheduler': scheduler if scheduler else None,
                    'metadata': metadata}
            if valid_data is None and saving_params.get("save_every_n_epochs", None) is None and saving_params.get("save_every_n_steps", None) is None:
                raise ValueError("If no validation data is provided, please specify 'save_every_n_epochs' and/or 'save_every_n_steps' in 'saving_params' to determine when to save the model.")
            # --- Save model (best) ----------------------------------
            if save_model_or_not:
                path_save_final = f"{self.model_path[:-3]}_best.pt"
                save_model(path=os.path.expanduser(path_save_final), **kwargs_save)
                log_msg += " | 💾 Model saved (best)"
                epoch_best = epoch

            # --- Save model also every n epochs? ----------------------------------
            if saving_params.get("save_every_n_epochs", None) is not None:
                save_epoch_interval = epoch - epoch_ref
                if save_epoch_interval >= saving_params["save_every_n_epochs"]:
                    path_save_per_epoch = f"{self.model_path[:-3]}_epoch{epoch}.pt"
                    save_model(path=os.path.expanduser(path_save_per_epoch), **kwargs_save)
                    log_msg += " | 💾 Model saved (epoch)"
                    epoch_ref = epoch

            # --- Save model also every n steps? ----------------------------------
            if self.Mlflow is not None:
                if saving_params.get("save_every_n_steps", None) is not None:
                    save_step_interval = global_step - step_ref
                    if save_step_interval >= saving_params["save_every_n_steps"]:
                        path_save_per_step = f"{self.model_path[:-3]}_step{global_step}.pt"
                        save_model(path=os.path.expanduser(path_save_per_step), **kwargs_save)
                        log_msg += " | 💾 Model saved (step)"
                        step_ref = global_step

            # --------------- MLFLOW --------------------------------------------------------
            # --- Save model also every n epochs (mlflow)? ----------------------------------
            if self.Mlflow is not None:
                if self.Mlflow_save_checkpoint_every_n_epochs is not None:
                    mlflow_save_epoch_interval = epoch - epoch_ref_mlflow
                    if mlflow_save_epoch_interval >= self.Mlflow_save_checkpoint_every_n_epochs:
                        path_save_mlflow = f"{self.model_path[:-3]}_epoch{epoch}_mlflow.pt"
                        save_model(path=os.path.expanduser(path_save_mlflow), **kwargs_save)
                        mlflow.log_artifact(path_save_mlflow, artifact_path="checkpoints")   
                        log_msg += " | 💾 Model saved (mlflow)"
                        epoch_ref_mlflow = epoch

            # --- Compute diagnostics (mlflow)? ----------------------------------
            if self.Mlflow is not None:
                if self.Mlflow_compute_diagnostics_every_n_epochs is not None:

                    ## Init downscaler
                    if epoch == 0:
                        path_save_mlflow = f"{self.model_dir}/modelPlaceholder_mlflow.pt"
                        save_model(path=os.path.expanduser(path_save_mlflow), **kwargs_save) # Save a model that contains all the metadata necessary to init properly downscaler
                        runner = self.d4p_func(id_dir=self.id_dir, input_data=self.input_data, forcing_data=self.forcing_data, model_file="modelPlaceholder_mlflow.pt", graph=self.graph_loc) # Run init
                        # print("🌐 (Mlflow) D4P DOWNSCALER READY ")

                    ## Determine if diagnostics are computed in this epoch
                    mlflow_diagnostic_epoch_interval = epoch - epoch_ref_mlflow_diagnostic
                    if mlflow_diagnostic_epoch_interval >= self.Mlflow_compute_diagnostics_every_n_epochs:

                        ## Predict and postprocess prediction
                        model.eval()
                        prd_mlflow = runner.downscale(model=model, return_pred=True, verbose=False)
                        # print(f"Pred (mlflow): {prd_mlflow}")
                        # print(f"Target (mlflow): {self.tgt_mlflow}")

                        ## Log scalars ------------------------------------------------------------------------------
                        Mlflow_scalars = self.Mlflow_diagnostics.get("scalars", None)
                        if Mlflow_scalars is not None:
                            mlflow_scalars_logs(tgt=self.tgt_mlflow, prd=prd_mlflow, vars=self.metadata_dict["vars_y"], mlflow_info=Mlflow_scalars, epoch=epoch)

                        ## Log figures ------------------------------------------------------------------------------
                        Mlflow_figures = self.Mlflow_diagnostics.get("figures", None)
                        if Mlflow_figures is not None:
                            if not Mlflow_figures.get("on_best", False):
                                mlflow_figures_logs(tgt=self.tgt_mlflow, prd=prd_mlflow, vars=self.metadata_dict["vars_y"], mlflow_info=Mlflow_figures, epoch=epoch)

                        ## Log scalars (xai) ------------------------------------------------------------------------------
                        Mlflow_scalars_xai = self.Mlflow_diagnostics.get("xai_scalars", None)
                        if Mlflow_scalars_xai is not None:
                            # mlflow_scalars_xai_logs(tgt=self.tgt_mlflow, prd=prd_mlflow, vars=self.metadata_dict["vars_y"], mlflow_info=Mlflow_scalars_xai, epoch=epoch)
                            print("XAI SCALARS LOGS not implemented ...")
    
                        ## Update epoch ref
                        epoch_ref_mlflow_diagnostic = epoch

            # --- Print the log -----------------------------------------------------
            print(log_msg)

        # --- Save best model to Mlflow and log figures (optional) ---
        if self.Mlflow is not None:
            ## Save best model ---
            if self.Mlflow.get("save_best", False):
                mlflow.log_artifact(path_save_final, artifact_path="checkpoints")   
            ## Log figures ---
            Mlflow_figures = self.Mlflow_diagnostics.get("figures", None)
            if Mlflow_figures is not None:
                if Mlflow_figures.get("on_best", False):
                    # Predict
                    runner = self.d4p_func(id_dir=self.id_dir, input_data=self.input_data, forcing_data=self.forcing_data, model_file=f"{self.model_save_name}_best.pt", graph=self.graph_loc) # Run init
                    prd_mlflow = runner.downscale(return_pred=True, verbose=False)
                    # Log figures
                    mlflow_figures_logs(tgt=self.tgt_mlflow, prd=prd_mlflow, vars=self.metadata_dict["vars_y"], mlflow_info = Mlflow_figures, epoch=epoch_best)
            
        # --- Return losses ---
        print("✅ Training completed successfully!")
        return train_losses, valid_losses if valid_losses else None
    
    # -------------------------------------------------------------------------
    def train(self, train_dataloader, valid_dataloader):
        """
        High-level method to start training using the training loop.
        
        Purpose:
            Calls the training loop, handles MLflow run ending, and prints completion message.
        
        Parameters:
            train_dataloader: Training DataLoader.
            valid_dataloader: Validation DataLoader.
        Returns:
            tuple: (train_loss, val_loss)
        """
        print(f"✅ CONFIGURATION READY FOR: {self.model_save_name}")   
        train_loss, val_loss = self.training_loop( 
                            model=self.model, 
                            train_data=train_dataloader, 
                            valid_data=valid_dataloader,
                            loss_function=self.loss_function,
                            training_params=self.training_params,
                            saving_params=self.saving_params,
                            device=self.device,
                            ema_decay=self.training_params.get("ema_decay", None),
                            metadata=self.metadata_dict,
                            kwargs=self.kwargs_training)
        
        # --- End Mlflow ---
        if self.Mlflow is not None:
            mlflow.end_run()

        print("----------------------------------------------------")
        print(f"✅ 🎯 {self.model_save_name}: Training finished successfully! 🎯 ✅")
        return train_loss, val_loss