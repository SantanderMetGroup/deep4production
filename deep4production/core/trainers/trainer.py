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

## Deep4production
from deep4production.deep.utils import EMA
from deep4production.deep.utils import save_model, resume_model
from deep4production.deep.preprocessing.normalizer import InputNormalizer
from deep4production.utils.general import get_func_from_string
from deep4production.utils.monitors import build_monitor
from deep4production.utils.log import get_logger
from deep4production.utils.paths import models_dir, aux_dir as aux_dir_for
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
        tracker=None,
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
        self.model_dir = models_dir(id_dir)
        self.aux_dir = aux_dir_for(id_dir)

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

        # --- Monitoring backend (MLflow or d4p-tracker) ---------------------
        # MLflow and d4p-tracker are mutually exclusive (enforced in
        # cli/train.py). The active backend is wrapped in a single ``Monitor``
        # so the training loop never branches on which one is in use; non-main
        # ranks get a no-op monitor so monitoring I/O stays on rank 0.
        self._monitor_runner = None  # lazily built on the first diagnostic epoch
        self.monitor = build_monitor(
            Mlflow, tracker, id_dir=id_dir, is_main=self._is_main
        )
        # Predictions over the validation period are only needed when the active
        # backend logs diagnostics; set up that machinery once here.
        if self.monitor.needs_predictions:
            self._setup_monitor_inputs(Mlflow if Mlflow is not None else tracker, data)

    # -------------------------------------------------------------------------
    def _setup_monitor_inputs(self, cfg, data):
        """
        Resolve the d4p downscaler function and the validation-period input /
        forcing specs used to produce predictions for monitoring diagnostics.

        Shared by the MLflow and d4p-tracker backends (they are mutually
        exclusive, so only one ever calls this) so the prediction machinery is
        defined in a single place.
        """
        d4p_name = cfg.get("func_name", "downscaler")
        d4p_module = cfg.get(
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
    def _monitor_predict(self, model, kwargs_save):
        """
        Run a validation-period prediction for monitoring diagnostics.

        The d4p downscaler ``runner`` is built lazily on the first call (it
        needs a checkpoint that carries the metadata) and cached for reuse on
        subsequent epochs. Returns the prediction xarray.Dataset; the matching
        ground truth lives in ``self.tgt_monitor``.
        """
        if self._monitor_runner is None:
            path_placeholder = f"{self.model_dir}/modelPlaceholder_monitor.pt"
            save_model(path=os.path.expanduser(path_placeholder), **kwargs_save)
            self._monitor_runner = self.d4p_func(
                id_dir=self.id_dir,
                input_data=self.input_data,
                forcing_data=self.forcing_data,
                model_file="modelPlaceholder_monitor.pt",
                graph=self.graph_loc,
            )
        model.eval()
        return self._monitor_runner.downscale(
            model=unwrap_model(model), return_pred=True, verbose=False
        )

    # -------------------------------------------------------------------------
    def _predict_from_best(self):
        """
        Run a validation-period prediction loading the *best* checkpoint from
        disk (used by the MLflow on-best figures hook at the end of training).
        """
        runner = self.d4p_func(
            id_dir=self.id_dir,
            input_data=self.input_data,
            forcing_data=self.forcing_data,
            model_file=f"{self.model_save_name}_best.pt",
            graph=self.graph_loc,
        )
        return runner.downscale(return_pred=True, verbose=False)

    # -------------------------------------------------------------------------
    def _to_model_space(self, ds):
        """
        Forward-transform a physical-units predictand xarray.Dataset into the
        model's normalized [-1,1] space, reproducing the exact training transform:
        per-variable operator forward (``operator_y``, e.g. ``sqrt`` for pr/hurs)
        followed by the affine normalizer (``self.norm_y``). Used by tracker
        metrics declared with ``space: model`` so per-variable errors become
        dimensionless and comparable across variables.

        No-op when no predictand normalizer is configured (``self.norm_y is
        None``); in that case the fields are already in raw operator space.
        """
        if self.norm_y is None:
            return ds
        vars_y = self.metadata_dict["vars_y"]
        # Stack variables into (T, C, G) in channel order.
        arr = np.stack(
            [np.asarray(ds[v].values, dtype=np.float32) for v in vars_y], axis=1
        )
        # Operator forward per channel (e.g. sqrt for pr/hurs), if any — this
        # must precede the affine, exactly as in pydataset.preprocess.
        op_info = self.metadata_dict.get("operator_y", None)
        if op_info is not None:
            for c, v in enumerate(vars_y):
                op_name = op_info["operator_func_per_variable"].get(v)
                if op_name is not None:
                    op_fn = get_func_from_string(op_info["module"], op_name)
                    arr[:, c] = op_fn(arr[:, c])
        # Affine normalize (operator space -> [-1,1]) via the trainer's
        # InputNormalizer, on the normalizer's buffer device.
        device = next(self.norm_y.buffers()).device
        t = torch.from_numpy(arr).to(device)
        t = self.norm_y.transform(t, in_place=False, channel_dim=1)
        arr = t.detach().cpu().numpy()
        # Rebuild a Dataset preserving the original coords/dims.
        out = ds.copy()
        for c, v in enumerate(vars_y):
            out[v] = (ds[v].dims, arr[:, c])
        return out

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
            # Validation-period ground truth for monitoring diagnostics, needed
            # only when the active monitoring backend logs diagnostics.
            if self.monitor.needs_predictions:
                self.tgt_monitor = valid_dataset.get_target_samples()
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

        # Best-checkpoint bookkeeping (referenced by the end-of-training monitor
        # hook even if no best checkpoint is ever written).
        epoch_best = 0
        path_save_final = None

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

            # --- Adaptive per-channel loss-weight update (e.g. DWA) ---------------
            # Loss functions that adapt their per-channel weights from the epoch's
            # training losses (e.g. DWAWeightedMseLoss) refresh them here, once per
            # epoch. No-op for every other loss (guarded by hasattr).
            if hasattr(loss_function, "on_epoch_end"):
                loss_function.on_epoch_end(epoch=epoch)

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
                valid_losses.append(val_loss)

            # --- Monitoring: record epoch losses (backend-agnostic) -----------
            # No-op on non-main ranks; MLflow logs the scalars, d4p-tracker
            # refreshes its losses.csv (the figure renders with the snapshot).
            self.monitor.log_losses(
                epoch,
                train_losses,
                valid_losses if valid_data is not None else None,
            )

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
            # (Historically gated on an active MLflow run; preserved via the
            # monitor's ``logs_checkpoints`` flag.)
            if self.monitor.logs_checkpoints:
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

            # --- Save + log a checkpoint artifact at the backend's cadence -------
            if self.monitor.should_save_checkpoint(epoch):
                path_save_ckpt = f"{self.model_path[:-3]}_epoch{epoch}_mlflow.pt"
                if self._is_main:
                    save_model(path=os.path.expanduser(path_save_ckpt), **kwargs_save)
                    self.monitor.log_checkpoint(path_save_ckpt)
                log_msg += " | 💾 model saved (mlflow)"
            # Restore training weights after all saves so the optimizer keeps working.
            if ema is not None and self._is_main:
                ema.restore()
            # Make sure other ranks don't race ahead before rank-0 finishes
            # writing checkpoints / Mlflow artifacts.
            barrier()

            # --- Monitoring: validation diagnostics at the backend's cadence ----
            # The monitor decides whether this epoch is due; the prediction
            # (potentially expensive) only runs when it will be logged. The
            # downscaler runner is built lazily inside ``_monitor_predict``.
            self.monitor.maybe_log_diagnostics(
                epoch=epoch,
                vars=self.metadata_dict["vars_y"],
                tgt=getattr(self, "tgt_monitor", None),
                predict=lambda: self._monitor_predict(model, kwargs_save),
                to_model_space=self._to_model_space,
            )

            # --- Per-epoch summary line --------------------------------------------
            if self._is_main:
                log.info("%s", log_msg)

        # --- Monitoring: end-of-training hook (best checkpoint / on-best figures)
        self.monitor.on_training_end(
            epoch_best=epoch_best,
            vars=self.metadata_dict["vars_y"],
            tgt=getattr(self, "tgt_monitor", None),
            best_checkpoint_path=path_save_final,
            predict_from_best=self._predict_from_best,
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
            Calls the training loop, tears down the monitoring backend, and prints completion message.

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

        # --- Tear down the monitoring backend (e.g. end the MLflow run) ---
        self.monitor.close()

        log.info("%s: training finished successfully.", self.model_save_name)
        return train_loss, val_loss
