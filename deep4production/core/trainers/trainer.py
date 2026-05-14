## Load libraries
import os
import math
import time
import torch
import contextlib
import numpy as np
from functools import partial
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

## MLFlow
import mlflow
import mlflow.pytorch

# from mlflow.tracking import MlflowClient
# from mlflow.exceptions import MlflowException
## Deep4production
from deep4production.deep.utils import EMA
from deep4production.deep.utils import save_model, resume_model
from deep4production.deep.preprocessing.normalizer import InputNormalizer
from deep4production.utils.general import get_func_from_string
from deep4production.utils.mlflow import *
from deep4production.utils.log import get_logger
from deep4production.utils.distributed import (
    is_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    barrier,
    all_reduce_mean,
    unwrap_model,
)

log = get_logger("trainer")


##################################################################################################################################
class trainer:
    def __init__(
        self,
        data,
        dataloader,
        id_dir,
        model_info,
        graph=None,
        d4dpy={},
        Mlflow=None,
        normalizer_info_x=None,
        normalizer_info_y=None,
        normalizer_info_f=None,
        hardware=None,
    ):
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
        log.info("Starting d4p trainer")
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
        if d4dpy:  # Is d4dpy dict not empty?
            self.pydataset = get_func_from_string(d4dpy["module"], d4dpy["name"])
            self.d4dpy = d4dpy["kwargs"]
        else:
            self.pydataset = get_func_from_string(
                "deep4production.core.pydatasets.pydataset", "pydataset"
            )

        # --- Device + DDP awareness -----------------------------------------
        # When DDP is active each rank pins its local GPU; otherwise fall back
        # to plain "cuda"/"cpu" so single-GPU training behaves unchanged.
        self.hardware = hardware or {}
        self._distributed = is_distributed()
        self._world_size = get_world_size()
        self._rank = get_rank()
        self._is_main = is_main_process()
        if self._distributed:
            self._local_rank = int(
                os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0))
            )
            self.device = f"cuda:{self._local_rank}"
        else:
            self._local_rank = 0
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Coarse type ("cuda" or "cpu") for comparisons that should not
        # care about the per-rank device index.
        self.device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"

        # --- Mixed precision (AMP) setup ---
        # Activate via YAML: model_info.training_params.amp: true
        # Uses bf16 on Ampere+ (A100/H100) when available — no GradScaler needed
        # — and falls back to fp16 + GradScaler elsewhere.
        amp_cfg = self.training_params.get("amp", False)
        want_amp = bool(amp_cfg) and self.device_type == "cuda"
        amp_dtype = (
            torch.bfloat16
            if (want_amp and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        self._amp_enabled = want_amp
        self._amp_dtype = amp_dtype
        # GradScaler only needed for fp16; bf16 has the same exponent range as fp32.
        self._scaler = (
            torch.amp.GradScaler("cuda")
            if want_amp and amp_dtype == torch.float16
            else None
        )
        if want_amp:
            log.info(
                "AMP enabled (dtype=%s, scaler=%s)",
                amp_dtype,
                "on" if self._scaler else "off",
            )

        self.id_dir = id_dir
        self.model_dir = f"{id_dir}/models/"
        self.aux_dir = f"{id_dir}/aux_files/"

        # --- BUILD GRAPH ---------------------------------------
        self.kwargs_training = self.training_params.get("kwargs", {})
        self.graph_loc = {}
        if graph is not None:
            edge_index = get_func_from_string(
                module_string=graph["module"],
                func_string=graph["name"],
                kwargs=graph.get("kwargs", None),
            )
            self.graph_loc["path"] = "edge_index.pt"
            torch.save(edge_index, f"{self.aux_dir}/{self.graph_loc["path"]}")
            log.info("Graph ready: function %s from %s", graph["name"], graph["module"])
        else:
            self.graph_loc = None

        # --- LOSS FUNCTION ---------------------------------------
        self.loss_function = get_func_from_string(
            module_string=self.loss_params["module"],
            func_string=self.loss_params["name"],
            kwargs=self.loss_params.get("kwargs", None),
        )
        log.info(
            "Loss ready: %s from %s",
            self.loss_params["name"],
            self.loss_params["module"],
        )

        # --- MODEL ---------------------------------------
        self.model_save_name = model_info["saving_params"]["model_save_name"]
        self.model = get_func_from_string(
            module_string=self.model_params["module"],
            func_string=self.model_params["name"],
            kwargs=self.model_params.get("kwargs", None),
        )
        self.model_path = f"{self.model_dir}/{self.model_save_name}.pt"
        log.info(
            "Model ready: %s from %s",
            self.model_params["name"],
            self.model_params["module"],
        )

        # --- INPUT NORMALIZERS (GPU-side, replaces per-sample CPU loop) -------
        # Each is an nn.Module with persistent buffers; .to(device) at training
        # time, and round-trips through state_dict so inference normalization
        # matches training normalization without recomputation.
        self.normalizer_info_x = normalizer_info_x
        self.normalizer_info_y = normalizer_info_y
        self.normalizer_info_f = normalizer_info_f
        self.norm_x = None
        self.norm_y = None
        self.norm_f = None

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
            self.Mlflow_compute_diagnostics_every_n_epochs = Mlflow.get(
                "compute_diagnostics_every_n_epochs", None
            )
            self.Mlflow_save_checkpoint_every_n_epochs = Mlflow.get(
                "save_checkpoint_every_n_epochs", None
            )
            if self.Mlflow_diagnostics is not None:
                ## Get d4p_downscaler function
                d4p_name = Mlflow.get("func_name", "downscaler")
                d4p_module = Mlflow.get(
                    "func_module", "deep4production.core.downscalers.downscaler"
                )
                self.d4p_func = get_func_from_string(
                    module_string=d4p_module, func_string=d4p_name
                )
                self.input_data = {
                    "paths": data["predictors"]["paths"],
                    "years": data["validation_period"],
                    "load_in_memory": data["load_in_memory"],
                }
                if data.get("forcings", None) is not None:
                    self.forcing_data = {
                        "paths": data["predictands"]["paths"],
                        "years": data["validation_period"],
                        "load_in_memory": data["load_in_memory"],
                    }
                else:
                    self.forcing_data = None

    # -------------------------------------------------------------------------
    def _amp_ctx(self):
        """Return an autocast context (or nullcontext) depending on AMP state."""
        if self._amp_enabled:
            return torch.amp.autocast(device_type="cuda", dtype=self._amp_dtype)
        return contextlib.nullcontext()

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
        metadata_dict["loss_params"] = {
            k: v for k, v in self.loss_params.items() if k not in ["name", "module"]
        }
        metadata_dict["loss_params"]["name"] = self.loss_params["name"]
        metadata_dict["loss_params"]["module"] = self.loss_params["module"]
        ### Model parameters
        metadata_dict["model_params"] = {}
        metadata_dict["model_params"] = {
            k: v for k, v in self.model_params.items() if k not in ["name", "module"]
        }
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
        self.metadata_dict["vars_x"], self.metadata_dict["vars_y"] = (
            pydataset.get_vars()
        )
        ### Lagged info
        self.metadata_dict["num_lagged_x"], self.metadata_dict["num_lagged_y"] = (
            pydataset.get_lagged_info()
        )
        ### Spatial info
        self.metadata_dict["lats_y"], self.metadata_dict["lons_y"] = (
            pydataset.get_coords()
        )
        (
            self.metadata_dict["transform_to_2D_x"],
            self.metadata_dict["transform_to_2D_y"],
        ) = pydataset.get_transform2D()
        (
            self.metadata_dict["H_x"],
            self.metadata_dict["W_x"],
            self.metadata_dict["H_y"],
            self.metadata_dict["W_y"],
        ) = pydataset.get_spatial_dims()
        self.metadata_dict["G_x"], self.metadata_dict["G_y"] = (
            pydataset.get_num_gridpoints()
        )
        ### Operator parameters (cont.) — operators still live in pydataset
        if self.data["predictors"].get("operator", None) is not None:
            self.metadata_dict["operator_x"] = pydataset.get_operator_info(
                predictands=False
            )
        if self.data["predictands"].get("operator", None) is not None:
            self.metadata_dict["operator_y"] = pydataset.get_operator_info(
                predictands=True
            )
        ### Forcings (vars/operator still come from pydataset; normalizer comes
        ### from this trainer's kwargs, see below)
        vars_f, idx_vars_f, _, operator_f = pydataset.get_forcings_info()
        self.metadata_dict["vars_f"] = vars_f
        self.metadata_dict["idx_vars_f"] = idx_vars_f
        self.metadata_dict["operator_f"] = operator_f
        ### Build the per-channel InputNormalizer modules now that we know
        ### the variable order. The dicts ``self.normalizer_info_*`` were
        ### populated from the recipe directly (see cli/train.py); pydataset
        ### never sees them under the new wiring.
        ###
        ### After construction, ``cli/train.py`` resolves stats_transform / per-
        ### variable methods exactly as before via pydataset.get_data_info, so
        ### we still need to consult pydataset for the merged dict (it knows
        ### how to fill the kwargs from the reference Zarr's stats arrays).
        ### Trainer-side: we just take whatever pydataset built for us when
        ### ``cli/train.py`` passes the recipe dict — pydataset still produces
        ### a normalized normalizer_info dict via ``_resolve_normalizer_info``.
        if self.normalizer_info_x is not None:
            resolved_x = pydataset._resolve_normalizer_info(
                self.normalizer_info_x, self.metadata_dict["vars_x"], predictand=False
            )
            self.metadata_dict["normalizer_x"] = resolved_x
            self.norm_x = InputNormalizer(
                resolved_x, self.metadata_dict["vars_x"], channel_dim=1
            )
            log.info(
                "InputNormalizer (X) ready: %s",
                resolved_x.get("normalizer_func_per_variable"),
            )
        if self.normalizer_info_y is not None:
            resolved_y = pydataset._resolve_normalizer_info(
                self.normalizer_info_y, self.metadata_dict["vars_y"], predictand=True
            )
            self.metadata_dict["normalizer_y"] = resolved_y
            self.norm_y = InputNormalizer(
                resolved_y, self.metadata_dict["vars_y"], channel_dim=1
            )
            log.info(
                "InputNormalizer (Y) ready: %s",
                resolved_y.get("normalizer_func_per_variable"),
            )
        if self.normalizer_info_f is not None and vars_f is not None:
            resolved_f = pydataset._resolve_normalizer_info(
                self.normalizer_info_f, vars_f, predictand=False, forcing=True
            )
            self.metadata_dict["normalizer_f"] = resolved_f
            self.norm_f = InputNormalizer(resolved_f, vars_f, channel_dim=1)
            log.info(
                "InputNormalizer (F) ready: %s",
                resolved_f.get("normalizer_func_per_variable"),
            )
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
        kwargs_pydataset = {
            "predictors": self.data["predictors"],
            "predictands": self.data["predictands"],
            "forcings": self.data.get("forcings", {}),
            "load_in_memory": self.data.get("load_in_memory", True),
            "cache_mb": self.data.get("zarr_cache_mb", None),
        }
        kwargs_pydataset.update(**self.d4dpy)
        train_dataset = self.pydataset(
            temporal_period=self.data["training_period"], **kwargs_pydataset
        )
        valid_dataset = None
        if self.data.get("validation_period", None) is not None:
            valid_dataset = self.pydataset(
                temporal_period=self.data["validation_period"], **kwargs_pydataset
            )
            if self.Mlflow is not None:
                if self.Mlflow_diagnostics is not None:
                    self.tgt_mlflow = valid_dataset.get_target_samples()
        ### Update metadata and save it with the new information
        self.metadata_dict = self.cont_metadata(train_dataset)
        # self.save_metadata(self.metadata_path)
        log.info("Pydatasets ready")
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
            log.warning("Number of workers not specified in YAML; using num_workers=0")
        shuffle = self.dataloader.get("shuffle", False)
        if self.dataloader.get("shuffle", None) is None:
            log.warning("Shuffle not specified in YAML; using shuffle=False")
        batch_size = self.dataloader.get("batch_size", 1)
        if self.dataloader.get("batch_size", None) is None:
            log.warning("Batch size not specified in YAML; using batch_size=1")
        kwargs_dataloader = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
        }
        # Pin host memory so tensors can be asynchronously copied to the GPU
        # via .to(device, non_blocking=True); only meaningful on CUDA.
        if self.device_type == "cuda":
            kwargs_dataloader["pin_memory"] = self.dataloader.get("pin_memory", True)
        # Keep workers alive between epochs to avoid re-forking them each time
        # (also preserves any per-worker in-memory data caches).
        if num_workers > 0:
            kwargs_dataloader["persistent_workers"] = self.dataloader.get(
                "persistent_workers", True
            )
            # PyTorch's prefetch_factor is meaningless when num_workers=0; only
            # attach it when at least one worker exists. Defaults to PyTorch's
            # built-in (2) when not set in YAML.
            if "prefetch_factor" in self.dataloader:
                kwargs_dataloader["prefetch_factor"] = self.dataloader[
                    "prefetch_factor"
                ]
        ## Create DataLoaders
        if self.graph is not None:
            DL = PyGDataLoader
        else:
            DL = TorchDataLoader

        # --- DDP: shard the dataset across ranks via DistributedSampler -------
        # DistributedSampler is incompatible with DataLoader's shuffle arg, so
        # we hand shuffling to the sampler. set_epoch() is called from the
        # training loop so the shuffle order varies across epochs.
        if self._distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self._world_size,
                rank=self._rank,
                shuffle=shuffle,
                drop_last=False,
            )
            kwargs_dataloader["sampler"] = train_sampler
            kwargs_dataloader["shuffle"] = False
            train_dataloader = DL(train_dataset, **kwargs_dataloader)
            valid_dataloader = None
            if valid_dataset is not None:
                valid_kwargs = {**kwargs_dataloader}
                valid_kwargs["sampler"] = DistributedSampler(
                    valid_dataset,
                    num_replicas=self._world_size,
                    rank=self._rank,
                    shuffle=False,
                    drop_last=False,
                )
                valid_dataloader = DL(valid_dataset, **valid_kwargs)
            if self._is_main:
                log.info("Dataloaders ready (DDP: world_size=%d)", self._world_size)
            return train_dataloader, valid_dataloader

        train_dataloader = DL(train_dataset, **kwargs_dataloader)
        valid_dataloader = None
        if valid_dataset is not None:
            valid_dataloader = DL(valid_dataset, **kwargs_dataloader)
        log.info("Dataloaders ready")
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
    def _normalize_inputs(
        self, x=None, y=None, f=None, channel_dim_x=1, channel_dim_y=1, channel_dim_f=1
    ):
        """
        Apply the GPU-side InputNormalizer modules in place to whichever of
        ``x``, ``y``, ``f`` are provided. Tensors must already be on
        ``self.device``. Returns the (possibly normalized) tensors in the same
        order. If a given normalizer is None, the corresponding tensor is
        returned unchanged.
        """
        if x is not None and self.norm_x is not None:
            x = self.norm_x(x, channel_dim=channel_dim_x)
        if y is not None and self.norm_y is not None:
            y = self.norm_y(y, channel_dim=channel_dim_y)
        if f is not None and self.norm_f is not None:
            f = self.norm_f(f, channel_dim=channel_dim_f)
        return x, y, f

    # -------------------------------------------------------------------------
    def model_backprop(
        self,
        model,
        data,
        optimizer,
        loss_function,
        device,
        is_this_training=True,
        **kwargs,
    ):
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
        non_blocking = self.device_type == "cuda"
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)

        if f[0] != "N/A":
            f = f.to(device, non_blocking=non_blocking)
            f_is_real = True
        else:
            B, Cy, *spatial = y.shape
            f = torch.zeros(B, Cy, *spatial, device=device)
            f_is_real = False

        # --- GPU-side normalization (replaces per-sample CPU loop in pydataset) ---
        # Only normalize the forcing tensor when it carries real data; the
        # zeros sentinel for "no forcing" stays as zeros.
        x, y, _ = self._normalize_inputs(x=x, y=y)
        if f_is_real:
            _, _, f = self._normalize_inputs(f=f)

        # --- Zero grads first (outside autocast) ---
        optimizer.zero_grad(set_to_none=True)

        # --- Forward + loss under autocast when AMP is enabled ---
        with self._amp_ctx():
            prediction = model(x, f)
            loss = loss_function(target=y, output=prediction)

        # --- Backpropagation ---
        if is_this_training:
            if self._scaler is not None:
                self._scaler.scale(loss).backward()
            else:
                loss.backward()

        # Return the detached loss tensor — the training loop accumulates
        # without calling .item() on every batch (avoids per-step GPU sync).
        return loss.detach()

    # -------------------------------------------------------------------------
    def update_params(self, optimizer, lr, scheduler=None):
        """
        Updates optimizer and scheduler, returns new learning rate.
        Purpose: Steps optimizer and scheduler, updates learning rate for training loop.
        Parameters:
            optimizer: PyTorch optimizer.
            lr: Current learning rate.
            scheduler: Learning rate scheduler (optional).
        Returns:
            float: Updated learning rate.
        """
        # --- Update optimizer ---
        optimizer.step()
        # --- Update scheduler ---
        if scheduler is not None:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
        # --- Return ---
        return lr

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
        grad_clip = training_params.get(
            "grad_clip", None
        )  # L2-norm clip applied after backward()

        # --- Model to device ------------------------------------------------
        model = model.to(device)
        # GPU-side normalizers travel with the model.
        if self.norm_x is not None:
            self.norm_x = self.norm_x.to(device)
        if self.norm_y is not None:
            self.norm_y = self.norm_y.to(device)
        if self.norm_f is not None:
            self.norm_f = self.norm_f.to(device)
        model_size = sum(p.numel() for p in model.parameters())
        model_mb = model_size * 4 / (1024**2)  # float32 = 4 bytes
        if self._is_main:
            log.info("Model parameters: %s (%.2f MB)", f"{model_size:,}", model_mb)

        # --- DDP wrap (after .to(device) so the buckets live on the right GPU) ---
        # find_unused_parameters defaults to False — flip it via
        # training_params.ddp_find_unused_parameters: true for graphs where
        # not every forward touches every parameter.
        if self._distributed:
            find_unused = bool(training_params.get("ddp_find_unused_parameters", False))
            model = DDP(
                model,
                device_ids=[self._local_rank],
                output_device=self._local_rank,
                find_unused_parameters=find_unused,
            )

        # --- torch.compile (optional) ----------------------------------------
        # Triggered by `training_params.compile: true` (default mode) or a
        # mode string e.g. "reduce-overhead" (CUDA graphs, best for fixed-shape
        # small batches) or "max-autotune" (aggressive, very long first epoch).
        # The compiled model exposes the same .state_dict() interface, so
        # checkpointing is unaffected. First epoch will be slow (kernel
        # compilation); subsequent epochs get the full speedup.
        compile_cfg = training_params.get("compile", False)
        if compile_cfg:
            if not hasattr(torch, "compile"):
                log.warning(
                    "torch.compile requires PyTorch >= 2.0; skipping (upgrade to benefit)."
                )
            else:
                compile_mode = (
                    compile_cfg if isinstance(compile_cfg, str) else "default"
                )
                model = torch.compile(model, mode=compile_mode)
                log.info(
                    "Model compiled (mode=%s); first epoch will be slow (kernel compilation).",
                    compile_mode,
                )

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
        global_step = 0  # Number of samples processed so far during training
        epoch_ref = 0  # Relevant for saving model every n epochs
        step_ref = 0  # Relevant for saving model every n steps

        # --- Learning rate scheduler ------------------------------------------
        scheduler = None
        scheduler_params = training_params.get("scheduler_params", None)
        if scheduler_params is not None:
            # Get scheduler function selected in YAML from "torch.optim.lr_scheduler"
            scheduler_type = scheduler_params["type"]
            scheduler_func = get_func_from_string(
                module_string="torch.optim.lr_scheduler", func_string=scheduler_type
            )
            scheduler_kwargs = scheduler_params.get("kwargs", None)
            # Handle LambdaLR separately (needs a callable)
            if scheduler_type == "LambdaLR":
                lambda_name = scheduler_params.get("lr_lambda", None)
                if lambda_name is None:
                    raise ValueError(
                        "LambdaLR requires 'lr_lambda' parameter in config YAML"
                    )
                lr_lambda_func = get_func_from_string(
                    module_string="deep4production.deep.schedulers",
                    func_string=lambda_name,
                )
                lr_lambda = partial(
                    lr_lambda_func, **scheduler_kwargs
                )  # Use functools.partial to freeze parameters
                # Instantiate scheduler properly
                scheduler = scheduler_func(optimizer, lr_lambda=lr_lambda)
            else:  # All other schedulers
                scheduler = scheduler_func(optimizer, **scheduler_kwargs)
            log.info("Loaded scheduler: %s", scheduler_type)

        # --- Resume training from a pretrained checkpoint? ------------------------------------------
        epoch_init = 0
        if saving_params.get("resume_checkpoint", None) is not None:
            path_checkpoint = f"{self.model_dir}/{saving_params["resume_checkpoint"]}"
            if os.path.exists(path_checkpoint):
                if self._is_main:
                    log.info("Resuming training from checkpoint: %s", path_checkpoint)
                # Load into the unwrapped module so the saved (DDP-stripped)
                # state_dict keys match.
                checkpoint = resume_model(
                    path=path_checkpoint,
                    model=unwrap_model(model),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                )
                epoch_init = epoch_ref = epoch = checkpoint["epoch"]
                step_ref = global_step = checkpoint["global_step"]
                train_losses = checkpoint.get("train_losses", [])
                valid_losses = checkpoint.get("valid_losses", [])
                valid_losses_arr = np.array(valid_losses)
                best_val_loss = np.min(valid_losses_arr)
                epoch_best_val_loss = np.where(valid_losses_arr == best_val_loss)[0][0]
                early_stopping_counter = epoch - epoch_best_val_loss
                if self._is_main:
                    log.info(
                        "Resume training: checkpoint=%s epoch=%d global_step=%d",
                        path_checkpoint,
                        epoch,
                        global_step,
                    )
            else:
                if self._is_main:
                    log.warning(
                        "Checkpoint specified for resuming training not found at %s; starting from scratch.",
                        path_checkpoint,
                    )

        # --- Ensemble Model Averaging (EMA) parameters ------------------------------------------
        # Build EMA against the underlying module so shadow keys do not carry
        # the DDP ``module.`` prefix; this also keeps EMA cheap (only one rank
        # would actually need it, but identical inputs → identical outputs).
        ema = None
        if ema_decay is not None:
            ema = EMA(unwrap_model(model), decay=ema_decay, device=device)

        # --- Mlflow counter ---
        epoch_ref_mlflow = 0
        epoch_ref_mlflow_diagnostic = 0

        # --- Loop over epochs ------------------------------------------
        if self._is_main:
            world_tag = (
                f" (DDP, world_size={self._world_size})" if self._distributed else ""
            )
            log.info(
                "Starting training for %d epochs on %s%s",
                num_epochs,
                str(device).upper(),
                world_tag,
            )
        for epoch in range(epoch_init, num_epochs):
            epoch_start = time.time()

            # --- DDP: re-seed the sampler so each rank gets a different shard
            # of the shuffled order at every epoch.
            if self._distributed and hasattr(train_data, "sampler"):
                sampler = train_data.sampler
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(epoch)

            # -----------------------------------------------------------------------------------------
            # --- Training phase: Loop over batches ---------------------------------------------------
            num_batches = len(train_data)
            model.train()
            # Accumulate loss as a tensor on-device so we don't sync to the CPU
            # every batch (.item() forces a GPU→CPU wait). One .item() call per
            # epoch below is enough.
            train_loss_sum = torch.zeros((), device=device)
            for batch_data in train_data:
                batch_loss = self.model_backprop(
                    model=model,
                    data=batch_data,
                    optimizer=optimizer,
                    loss_function=loss_function,
                    device=device,
                    is_this_training=True,
                    **kwargs,
                )
                # Accept either a tensor (new base/SongUNet path) or a scalar
                # (subclasses still returning .item()) — keep accumulation generic.
                if not torch.is_tensor(batch_loss):
                    batch_loss = torch.as_tensor(batch_loss, device=device)
                train_loss_sum = train_loss_sum + batch_loss.detach()

                # --- Gradient clipping + optimizer step (AMP-aware) ---
                if self._scaler is not None:
                    # Unscale gradients in place before clipping so clip_grad_norm_
                    # sees real gradient magnitudes.
                    if grad_clip is not None:
                        self._scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    self._scaler.step(optimizer)
                    self._scaler.update()
                    if scheduler is not None:
                        scheduler.step()
                        current_lr = scheduler.get_last_lr()[0]
                else:
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    # --- Scheduler: Update learning rate, and optimizer and loss ---
                    current_lr = self.update_params(
                        optimizer=optimizer, lr=current_lr, scheduler=scheduler
                    )

                # --- Update EMA per step (required for diffusion models with high decay) ---
                if ema is not None:
                    ema.update()

                global_step += 1

            # --- Store training loss ---
            train_loss_local = train_loss_sum / num_batches
            # All-reduce average across ranks so every process has the same
            # epoch loss (drives early stopping and logging consistently).
            train_loss_local = all_reduce_mean(train_loss_local)
            train_loss = train_loss_local.item()  # one sync per epoch
            train_losses.append(train_loss)
            if self.Mlflow is not None:
                mlflow.log_metric("train_loss_epoch", train_loss, step=int(epoch))

            # -----------------------------------------------------------------------------------------
            # --- Validation phase: Loop over batches -------------------------------------------------
            val_loss = None
            if valid_data is not None:
                model.eval()
                with torch.no_grad():
                    num_batches = len(valid_data)
                    val_loss_sum = torch.zeros((), device=device)
                    for batch_data in valid_data:
                        batch_loss = self.model_backprop(
                            model=model,
                            data=batch_data,
                            optimizer=optimizer,
                            loss_function=loss_function,
                            device=device,
                            is_this_training=False,
                            **kwargs,
                        )
                        if not torch.is_tensor(batch_loss):
                            batch_loss = torch.as_tensor(batch_loss, device=device)
                        val_loss_sum = val_loss_sum + batch_loss.detach()
                    val_loss_local = val_loss_sum / num_batches
                    val_loss_local = all_reduce_mean(val_loss_local)
                    val_loss = val_loss_local.item()
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
                        if self._is_main:
                            log.info(
                                "[%s] Early stopping triggered after %d epochs.",
                                timestamp,
                                epoch,
                            )
                        break

            # --- Save model (general info & checks) ----------------------------------
            # If EMA is active, apply shadow weights before ALL save calls so every
            # checkpoint contains the temporally-smoothed weights (crucial for
            # diffusion samplers). Restore training weights afterwards so the
            # optimizer can continue using them.
            # In DDP only rank 0 writes to disk; all ranks share filesystem and
            # parameters are synchronized, so a single writer is sufficient.
            if ema is not None and self._is_main:
                ema.apply_shadow()
            # Always save the *unwrapped* module so the resulting state_dict
            # has no ``module.`` prefix and can be loaded by inference code
            # (or single-GPU training) without modification.
            kwargs_save = {
                "epoch": epoch,
                "global_step": global_step,
                "train_losses": train_losses,
                "valid_losses": valid_losses if valid_data else None,
                "model": unwrap_model(model),
                "optimizer": optimizer,
                "scheduler": scheduler if scheduler else None,
                "metadata": metadata,
            }
            if (
                valid_data is None
                and saving_params.get("save_every_n_epochs", None) is None
                and saving_params.get("save_every_n_steps", None) is None
            ):
                raise ValueError(
                    "If no validation data is provided, please specify 'save_every_n_epochs' and/or 'save_every_n_steps' in 'saving_params' to determine when to save the model."
                )
            # --- Save model (best) ----------------------------------
            if save_model_or_not:
                path_save_final = f"{self.model_path[:-3]}_best.pt"
                if self._is_main:
                    save_model(path=os.path.expanduser(path_save_final), **kwargs_save)
                log_msg += " | 💾 model saved (best)"
                epoch_best = epoch

            # --- Save model also every n epochs? ----------------------------------
            if saving_params.get("save_every_n_epochs", None) is not None:
                save_epoch_interval = epoch - epoch_ref
                if save_epoch_interval >= saving_params["save_every_n_epochs"]:
                    path_save_per_epoch = f"{self.model_path[:-3]}_epoch{epoch}.pt"
                    if self._is_main:
                        save_model(
                            path=os.path.expanduser(path_save_per_epoch), **kwargs_save
                        )
                    log_msg += " | 💾 model saved (epoch)"
                    epoch_ref = epoch

            # --- Save model also every n steps? ----------------------------------
            if self.Mlflow is not None:
                if saving_params.get("save_every_n_steps", None) is not None:
                    save_step_interval = global_step - step_ref
                    if save_step_interval >= saving_params["save_every_n_steps"]:
                        path_save_per_step = (
                            f"{self.model_path[:-3]}_step{global_step}.pt"
                        )
                        if self._is_main:
                            save_model(
                                path=os.path.expanduser(path_save_per_step),
                                **kwargs_save,
                            )
                        log_msg += " | 💾 model saved (step)"
                        step_ref = global_step

            # --------------- MLFLOW --------------------------------------------------------
            # --- Save model also every n epochs (mlflow)? ----------------------------------
            if self.Mlflow is not None:
                if self.Mlflow_save_checkpoint_every_n_epochs is not None:
                    mlflow_save_epoch_interval = epoch - epoch_ref_mlflow
                    if (
                        mlflow_save_epoch_interval
                        >= self.Mlflow_save_checkpoint_every_n_epochs
                    ):
                        path_save_mlflow = (
                            f"{self.model_path[:-3]}_epoch{epoch}_mlflow.pt"
                        )
                        if self._is_main:
                            save_model(
                                path=os.path.expanduser(path_save_mlflow), **kwargs_save
                            )
                            mlflow.log_artifact(
                                path_save_mlflow, artifact_path="checkpoints"
                            )
                        log_msg += " | 💾 model saved (mlflow)"
                        epoch_ref_mlflow = epoch
            # Restore training weights after all saves so the optimizer keeps working.
            if ema is not None and self._is_main:
                ema.restore()
            # Make sure other ranks don't race ahead before rank-0 finishes
            # writing checkpoints / Mlflow artifacts.
            barrier()

            # --- Compute diagnostics (mlflow)? ----------------------------------
            if self.Mlflow is not None:
                if self.Mlflow_compute_diagnostics_every_n_epochs is not None:
                    ## Init downscaler
                    if epoch == 0:
                        path_save_mlflow = (
                            f"{self.model_dir}/modelPlaceholder_mlflow.pt"
                        )
                        save_model(
                            path=os.path.expanduser(path_save_mlflow), **kwargs_save
                        )  # Save a model that contains all the metadata necessary to init properly downscaler
                        runner = self.d4p_func(
                            id_dir=self.id_dir,
                            input_data=self.input_data,
                            forcing_data=self.forcing_data,
                            model_file="modelPlaceholder_mlflow.pt",
                            graph=self.graph_loc,
                        )  # Run init
                        # print("🌐 (Mlflow) D4P DOWNSCALER READY ")

                    ## Determine if diagnostics are computed in this epoch
                    mlflow_diagnostic_epoch_interval = (
                        epoch - epoch_ref_mlflow_diagnostic
                    )
                    if (
                        mlflow_diagnostic_epoch_interval
                        >= self.Mlflow_compute_diagnostics_every_n_epochs
                    ):
                        ## Predict and postprocess prediction
                        model.eval()
                        # Downscaler expects the unwrapped module (its forward
                        # signatures don't go through DDP's wrapper).
                        prd_mlflow = runner.downscale(
                            model=unwrap_model(model), return_pred=True, verbose=False
                        )
                        # print(f"Pred (mlflow): {prd_mlflow}")
                        # print(f"Target (mlflow): {self.tgt_mlflow}")

                        ## Log scalars ------------------------------------------------------------------------------
                        Mlflow_scalars = self.Mlflow_diagnostics.get("scalars", None)
                        if Mlflow_scalars is not None:
                            mlflow_scalars_logs(
                                tgt=self.tgt_mlflow,
                                prd=prd_mlflow,
                                vars=self.metadata_dict["vars_y"],
                                mlflow_info=Mlflow_scalars,
                                epoch=epoch,
                            )

                        ## Log figures ------------------------------------------------------------------------------
                        Mlflow_figures = self.Mlflow_diagnostics.get("figures", None)
                        if Mlflow_figures is not None:
                            if not Mlflow_figures.get("on_best", False):
                                mlflow_figures_logs(
                                    tgt=self.tgt_mlflow,
                                    prd=prd_mlflow,
                                    vars=self.metadata_dict["vars_y"],
                                    mlflow_info=Mlflow_figures,
                                    epoch=epoch,
                                )

                        ## Log scalars (xai) ------------------------------------------------------------------------------
                        Mlflow_scalars_xai = self.Mlflow_diagnostics.get(
                            "xai_scalars", None
                        )
                        if Mlflow_scalars_xai is not None:
                            # mlflow_scalars_xai_logs(tgt=self.tgt_mlflow, prd=prd_mlflow, vars=self.metadata_dict["vars_y"], mlflow_info=Mlflow_scalars_xai, epoch=epoch)
                            log.warning("XAI scalars logs not implemented; skipping.")

                        ## Update epoch ref
                        epoch_ref_mlflow_diagnostic = epoch

            # --- Per-epoch summary line --------------------------------------------
            if self._is_main:
                log.info("%s", log_msg)

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
                    runner = self.d4p_func(
                        id_dir=self.id_dir,
                        input_data=self.input_data,
                        forcing_data=self.forcing_data,
                        model_file=f"{self.model_save_name}_best.pt",
                        graph=self.graph_loc,
                    )  # Run init
                    prd_mlflow = runner.downscale(return_pred=True, verbose=False)
                    # Log figures
                    mlflow_figures_logs(
                        tgt=self.tgt_mlflow,
                        prd=prd_mlflow,
                        vars=self.metadata_dict["vars_y"],
                        mlflow_info=Mlflow_figures,
                        epoch=epoch_best,
                    )

        # --- Return losses ---
        if self._is_main:
            log.info("Training completed successfully.")
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
        log.info("Configuration ready for: %s", self.model_save_name)
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
            kwargs=self.kwargs_training,
        )

        # --- End Mlflow ---
        if self.Mlflow is not None:
            mlflow.end_run()

        log.info("%s: training finished successfully.", self.model_save_name)
        return train_loss, val_loss
