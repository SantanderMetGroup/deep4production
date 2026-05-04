import zarr
import torch
import numpy as np
import xarray as xr
from contextlib import nullcontext
from torch import from_numpy
## Deep4production
from deep4production.deep.utils import load_model
from deep4production.deep.preprocessing.normalizer import InputNormalizer
from deep4production.utils.trans import from_pred_to_xarray, compute_valid_mask
from deep4production.utils.general import get_func_from_string
from deep4production.utils.temporal import get_dates_from_yaml, get_sample_map, get_pairs
from deep4production.utils.zarr import open_zarr_store
from deep4production.utils.log import get_logger

log = get_logger("downscaler")


# Map YAML strings → torch dtypes for AMP autocast.
_AMP_DTYPES = {
    None:        None,
    "":          None,
    "none":      None,
    "float32":   torch.float32,
    "fp32":      torch.float32,
    "float16":   torch.float16,
    "fp16":      torch.float16,
    "half":      torch.float16,
    "bfloat16":  torch.bfloat16,
    "bf16":      torch.bfloat16,
}

##################################################################################################################################
class downscaler:
    """
    Downscaler class for applying trained models to input data and generating predictions.
    Purpose: Loads model and metadata, preprocesses input, handles forcings, and saves predictions.
    Parameters:
        id_dir (str): Experiment directory.
        input_data (dict): Input data configuration.
        model_file (str, optional): Model checkpoint filename.
        saving_info (dict, optional): Output saving configuration.
        ensemble_size (int, optional): Number of ensemble members.
        graph (dict, optional): Graph configuration for GNN models.
        forcing_data (dict, optional): Forcing data configuration.
    """
    def __init__(self, id_dir, input_data, model_file=None, saving_info=None, ensemble_size=1, graph=None, forcing_data=None):
        """
        Initializes the D4P Downscaler.
        """
        log.info("Starting d4p downscaler")
        # --- SELF PARAMETERS ---
        self.ensemble_size = ensemble_size
        self.graph = graph
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        log.debug("Device: %s", self.device)

        # --- GET MODEL AND METADATA FROM CHECKPOINT ---
        if model_file is not None:
            model_path = f"{id_dir}/models/{model_file}"
            self.model, self.metadata = load_model(path=model_path, map_location=self.device, return_metadata=True)
            self.model.to(self.device)
            log.info("Model and metadata loaded from %s", model_path)

        # --- OUTPUT PATH ---
        self.saving_info = saving_info
        if self.saving_info is not None:
            file = self.saving_info["file"]
            self.output_path = f"{id_dir}/predictions/{file}"
            log.info("Predictions will be saved at: %s", self.output_path)

        # --- GET INFO FROM METADATA ---
        self.update_self(
            input_data["paths"],
            fmt=input_data.get("format", "auto"),
            cache_mb=input_data.get("zarr_cache_mb", None),
        )

        # --- GET DOWNSCALING DATES ---
        freq = self.x[0].attrs["frequency"]
        dates_yaml = get_dates_from_yaml(input_data["years"], freq=freq)
        self.sample_map, dates = get_sample_map(dates_yaml, self.x)
        self.pairs = get_pairs(dates=dates, freq=freq, num_lagged_x=self.num_lagged_x)
        self.target_dates = list(self.pairs.keys())
        num_samples = len(self.pairs)
        log.info("Number of initialization dates: %d", num_samples)

        # --- LOAD INPUT DATA IN MEMORY? ---
        load_in_memory = input_data.get("load_in_memory", True)
        if load_in_memory: # If dataset fits in memory, load input data to speed up
            x_data = [np.array(x["data"]) for x in self.x]
            self.data = {"x": x_data}
            log.info("Predictor data loaded into memory.")
        else:
            self.data = {"x": [x["data"] for x in self.x]}

        # --- FORCINGS (optional) ---
        self.forcing_data = forcing_data
        if self.forcing_data is not None:
            self.update_self_with_forcings(
                forcing_data["paths"],
                fmt=forcing_data.get("format", "auto"),
                cache_mb=forcing_data.get("zarr_cache_mb", None),
            )
            freq = self.f[0].attrs["frequency"]
            dates_yaml = get_dates_from_yaml(forcing_data["years"], freq=freq)
            self.sample_map_f, _ = get_sample_map(dates_yaml, self.x)
            load_in_memory = input_data.get("load_in_memory", True)
            if load_in_memory: # If dataset fits in memory, load input data to speed up
                f_data = [np.array(f["data"]) for f in self.f]
                self.data.update({"f": f_data})
                log.info("Forcing data loaded into memory.")
            else:
                self.data.update({"f": [f["data"] for f in self.f]})

        # --- MAPPING TO XARRAY INFO ---
        ## Template
        if self.saving_info is not None:
            template_path = self.saving_info.get("template", None)
            if template_path is not None:
                self.template = xr.open_dataset(template_path)
                self.lats = None
                self.lons = None
                log.info("Using provided template for coordinates.")
            else:
                self.template = None
                self.lats = self.metadata["lats_y"]
                self.lons = self.metadata["lons_y"]
                log.warning("No template provided; using lats/lons from metadata.")
        else:
            self.template = None
            self.lats = self.metadata["lats_y"]
            self.lons = self.metadata["lons_y"]
            log.warning("No template provided; using lats/lons from metadata.")
        ## Formatting
        self.format_output = None
        if self.saving_info is not None:
            self.format_output = self.saving_info.get("formatting", None)
            if self.format_output is not None:
                formatting_module = "deep4production.utils.formatting"
                formatting_name = self.saving_info["formatting"]["name"]
                self.formatting_func = get_func_from_string(formatting_module, formatting_name)
                self.formatting_kwargs = self.saving_info["formatting"].get("kwargs", None)

        # --- BUILD GRAPH ---------------------------------------
        if self.graph is not None:
            if self.graph["path"] is not None:
                self.edge_index = torch.load(f"{id_dir}/aux_files/{self.graph['path']}", weights_only=False)
                log.info("Graph loaded from %s", self.graph['path'])
            else:
                self.edge_index = get_func_from_string(module_string=self.graph["module"],func_string=self.graph["name"], kwargs=self.graph.get("kwargs", None))
                torch.save(self.edge_index, f"{self.aux_dir}/aux_files/edge_index_B.pt")
                log.info("Graph ready: function %s from %s", self.graph['name'], self.graph['module'])

        # --- POSTPROCESS FUNC ---------------------------------------
        postprocess_module = "deep4production.deep.postprocessors"
        # Specific
        self.post_func_kwargs = {}
        if self.loss_params["name"] == "NLLBerGammaLoss":
            postprocess_name = "from_bergamma_to_pred"
            self.post_func_kwargs = {"threshold": self.loss_params["kwargs"]["threshold"]}
        elif self.loss_params["name"] == "NLLGaussianLoss":
            postprocess_name = "from_gaussian_to_pred"
        else:
            postprocess_name = "standard"
        self.post_func = get_func_from_string(postprocess_module, postprocess_name)

        # --- Pre-resolve operator callables ONCE (per-channel CPU funcs) ---
        # Operators stay on CPU because they're non-linear; normalization
        # moved to GPU-side InputNormalizer modules below.
        self._ops_x = self._build_operator_pipeline(self.operator_x, self.vars_x)
        self._ops_y = self._build_operator_pipeline(self.operator_y, self.vars_y)
        if forcing_data is not None:
            self._ops_f = self._build_operator_pipeline(self.operator_f, self.vars_f)
        else:
            self._ops_f = None

        # --- Build InputNormalizer modules from saved metadata --------
        # The metadata's normalizer dicts were resolved at training time by
        # pydataset._resolve_normalizer_info — kwargs / methods / stats_transform
        # are all baked in. Buffers travel with the module to GPU.
        self.norm_x = self._build_input_normalizer(self.normalizer_x, self.vars_x, channel_dim=1)
        self.norm_y = self._build_input_normalizer(self.normalizer_y, self.vars_y, channel_dim=1)
        self.norm_f = (
            self._build_input_normalizer(self.normalizer_f, self.vars_f, channel_dim=1)
            if forcing_data is not None else None
        )
        if self.norm_x is not None: self.norm_x = self.norm_x.to(self.device)
        if self.norm_y is not None: self.norm_y = self.norm_y.to(self.device)
        if self.norm_f is not None: self.norm_f = self.norm_f.to(self.device)

        # --- Pre-compute the template's spatial NaN mask ONCE -----------
        # `from_pred_to_xarray` computes this per call when given a template;
        # caching avoids redoing the isnull → mean → where chain for every
        # member's xarray build at the end of the run.
        self._template_mask = (
            compute_valid_mask(self.template) if self.template is not None else None
        )

        # --- Runtime/inference flags (set by downscale()) ---------------
        self._amp_dtype = None      # torch.dtype or None
        self._is_compiled = False   # set True after first torch.compile()
        self._cuda = (self.device == 'cuda')

    # ---------------------------------------------------------------------------------------------------------------------<
    @staticmethod
    def _build_operator_pipeline(operator_info, vars):
        """
        Resolve per-channel operator callables once.

        Returns
        -------
        ops : list[callable or None] or None
        """
        if operator_info is None:
            return None
        ops = []
        for var in vars:
            name = operator_info["operator_func_per_variable"].get(var)
            if name is None:
                ops.append(None)
            else:
                ops.append(get_func_from_string(operator_info["module"], name))
        return ops

    # ---------------------------------------------------------------------------------------------------------------------<
    @staticmethod
    def _build_input_normalizer(normalizer_info, vars, channel_dim=1):
        """
        Construct an InputNormalizer from a metadata-resolved normalizer dict
        (the same shape produced by pydataset._resolve_normalizer_info at
        training time and saved in the checkpoint metadata). Returns None if
        ``normalizer_info`` is None.
        """
        if normalizer_info is None:
            return None
        return InputNormalizer(normalizer_info, vars, channel_dim=channel_dim)

    # ---------------------------------------------------------------------------------------------------------------------<
    def update_self(self, paths, fmt="auto", cache_mb=None):
        """
        Updates internal attributes using input paths and metadata.
        Parameters:
            paths (list): List of Zarr file paths.
            fmt   (str):  "d4p", "anemoi", or "auto" — opens stores via
                          utils.zarr.open_zarr_store.
            cache_mb (int or None): If set, wrap each store in a
                          zarr.LRUStoreCache of the given megabyte budget.
        """
        # --- Files (X)---
        self.x = [open_zarr_store(p, fmt=fmt, cache_mb=cache_mb) for p in paths]
        # --- Variables --- 
        self.vars_y = self.metadata["vars_y"]
        self.vars_x = self.metadata["vars_x"]
        self.idx_vars_x = [self.x[0].attrs["variables"][var] for var in self.vars_x]
        self.num_lagged_x = self.metadata["num_lagged_x"]
        # --- Normalizer ---
        self.normalizer_x = self.metadata.get("normalizer_x", None)
        if self.normalizer_x is not None:
            log.debug("Normalizer (X): %s", self.normalizer_x.get("normalizer_func_per_variable"))
        # --- Denormalizer (Prediction) ---
        self.normalizer_y = self.metadata.get("normalizer_y", None)
        if self.normalizer_y is not None:
            log.debug("Denormalizer (Y): %s", self.normalizer_y.get("normalizer_func_per_variable"))
        # --- Operator ---
        self.operator_x = self.metadata.get("operator_x", None)
        if self.operator_x is not None:
            log.debug("Operator (X): %s", self.operator_x.get("operator_func_per_variable"))
        # --- Deoperator (Prediction) ---
        self.operator_y = self.metadata.get("operator_y", None)
        if self.operator_y is not None:
            log.debug("Deoperator (Y): %s", self.operator_y.get("operator_func_per_variable"))
        # --- Loss params --- 
        self.loss_params = self.metadata.get("loss_params", None)
        # --- Transform to 2D --- 
        self.transform_to_2D_x = self.metadata.get("transform_to_2D_x", False)
        self.transform_to_2D_y = self.metadata.get("transform_to_2D_y", False)
        # --- Input and output 2D spatial dimensions ---
        self.H_x, self.W_x = self.metadata.get("H_x", None), self.metadata.get("W_x", None)
        self.H_y, self.W_y = self.metadata.get("H_y", None), self.metadata.get("W_y", None)
        # --- Input and output expected number of gridpoints ---
        self.G_x = self.metadata.get("G_x", None)
        self.G_y = self.metadata.get("G_y", None)


    # ---------------------------------------------------------------------------------------------------------------------<
    def update_self_with_forcings(self, fpaths, fmt="auto", cache_mb=None):
        """
        Updates internal attributes for forcings using input paths and metadata.
        Parameters:
            fpaths (list): List of Zarr file paths for forcings.
            fmt    (str):  "d4p", "anemoi", or "auto".
            cache_mb (int or None): If set, wrap each store in an LRU cache.
        """
        # --- Forcings info ---
        self.f = [open_zarr_store(p, fmt=fmt, cache_mb=cache_mb) for p in fpaths]
        self.vars_f = self.metadata["vars_f"]
        self.idx_vars_f = [self.f[0].attrs["variables"][var] for var in self.vars_f]
        self.normalizer_f = self.metadata.get("normalizer_f", None)
        if self.normalizer_f is not None:
            log.debug("Normalizer (F): %s", self.normalizer_f.get("normalizer_func_per_variable"))
        self.operator_f = self.metadata.get("operator_f", None)
        if self.operator_f is not None:
            log.debug("Operator (F): %s", self.operator_f.get("operator_func_per_variable"))


    # ---------------------------------------------------------------------------------------------------------------------<
    def graphPredict(self, x, edge_index, model, f=["N/A"]):
        """
        Placeholder for graph prediction. Should be implemented in subclass for PyTorch Geometric models.
        Parameters:
            x (torch.Tensor): Input tensor.
            edge_index: Graph edge indices.
            model: PyTorch model.
            f: Forcing tensor (optional).
        Returns:
            np.ndarray: Prediction array.
        """
        assert False, (
            "🛑 Placeholder for the graphPredict function. Create a subclass of d4p_downscaler "
            "that implements graphPredict to convert the PyTorch data into a format compatible "
            "with PyTorch Geometric (PyG) graph objects."
        )

    # ---------------------------------------------------------------------------------------------------------------------<
    def preprocess(self, date, data, vars, idx_vars, sample_map, operator=None, transform_to_2D=False, H=None, W=None, ops=None, to_device=True):
        """
        Preprocess a sample: index → operator → reshape → to-tensor.

        Normalization no longer happens here — the downscaler applies
        ``self.norm_x`` (and ``self.norm_f``) on the GPU after the H2D
        transfer, mirroring the trainer's GPU-side path.

        Accepts two equivalent operator interfaces:
          - Fast path: pre-resolved callable list ``ops``.
          - Legacy path: dict-based ``operator`` (resolved on the fly).

        Returns
        -------
        torch.Tensor on ``self.device`` (when to_device=True) or CPU.
        """
        # -- Lazily resolve operator callables if only the legacy dict was given --
        if ops is None and operator is not None:
            ops = [
                get_func_from_string(operator["module"], operator["operator_func_per_variable"][v])
                if operator["operator_func_per_variable"].get(v) is not None else None
                for v in vars
            ]

        # -- Get sample --
        i, j = sample_map[date]
        x = data[i][j][idx_vars]                       # (C, G)

        # --- Operator (per-channel) ---
        if ops is not None:
            for c, fn in enumerate(ops):
                if fn is not None:
                    x[c, :] = fn(x[c, :])
        # --- Transform to 2D ---
        if transform_to_2D:
            C, G = x.shape
            x = x.reshape(C, H, W)
        # --- Convert to torch tensor ---
        x = from_numpy(x.copy())
        return x.to(self.device) if to_device else x

    # ---------------------------------------------------------------------------------------------------------------------<
    def _preprocess_single_date(self, target_date) -> torch.Tensor:
        """
        Preprocess one target date into an unbatched CPU tensor.
        Returns (C, H, W) for single-step models or (n_lag, C, H, W) for lagged ones.
        Caller is responsible for stacking and transferring to GPU once.
        """
        dates = self.pairs[target_date]
        if len(dates) > 1:
            return torch.stack([
                self.preprocess(date, self.data["x"], self.vars_x, self.idx_vars_x,
                                self.sample_map, ops=self._ops_x,
                                transform_to_2D=self.transform_to_2D_x, H=self.H_x, W=self.W_x,
                                to_device=False)
                for date in dates
            ])  # (n_lag, C, H, W) on CPU
        return self.preprocess(target_date, self.data["x"], self.vars_x, self.idx_vars_x,
                               self.sample_map, ops=self._ops_x,
                               transform_to_2D=self.transform_to_2D_x, H=self.H_x, W=self.W_x,
                               to_device=False)

    # ---------------------------------------------------------------------------------------------------------------------<
    def _preprocess_forcing_date(self, target_date) -> torch.Tensor:
        """
        Preprocess forcings for one date as an unbatched CPU tensor.
        Returns (C_y, H, W) or a CPU zeros tensor when no forcing data is configured.
        """
        if self.forcing_data is not None:
            return self.preprocess(target_date, self.data["f"], self.vars_f, self.idx_vars_f,
                                   self.sample_map_f, ops=self._ops_f,
                                   transform_to_2D=self.transform_to_2D_y, H=self.H_y, W=self.W_y,
                                   to_device=False)
        Cy = len(self.vars_y)
        spatial = [self.H_y, self.W_y] if self.transform_to_2D_y else [self.G_y]
        return torch.zeros(Cy, *spatial)  # CPU; caller transfers to device

    # ---------------------------------------------------------------------------------------------------------------------<
    def _stack_to_device(self, tensors_cpu) -> torch.Tensor:
        """
        Stack a list of CPU tensors and issue a single async H2D transfer.
        On CPU-only runs this just stacks (no-op transfer).
        """
        batch_cpu = torch.stack(tensors_cpu)
        if not self._cuda:
            return batch_cpu
        # pin_memory enables true async; cost is one allocation per batch.
        batch_cpu = batch_cpu.pin_memory()
        return batch_cpu.to(self.device, non_blocking=True)

    # ---------------------------------------------------------------------------------------------------------------------<
    @staticmethod
    def _parse_amp_dtype(amp_dtype):
        """Convert YAML-friendly amp_dtype (str/None/torch.dtype) into a torch.dtype or None."""
        if amp_dtype is None:
            return None
        if isinstance(amp_dtype, torch.dtype):
            return amp_dtype if amp_dtype != torch.float32 else None
        if isinstance(amp_dtype, str):
            return _AMP_DTYPES.get(amp_dtype.lower(), None)
        raise TypeError(f"amp_dtype must be str, None, or torch.dtype; got {type(amp_dtype)}")

    # ---------------------------------------------------------------------------------------------------------------------<
    def _amp_ctx(self):
        """Autocast context for mixed-precision inference, or nullcontext if disabled."""
        if self._amp_dtype is None or not self._cuda:
            return nullcontext()
        return torch.autocast(device_type='cuda', dtype=self._amp_dtype)

    # ---------------------------------------------------------------------------------------------------------------------<
    def _maybe_compile(self, model, compile_flag):
        """
        Wrap model with torch.compile on first invocation.
        dynamic=True so the last (smaller) batch does not trigger recompilation.
        Failures fall back to eager mode with a warning.
        """
        if not compile_flag or self._is_compiled:
            return model
        try:
            compiled = torch.compile(model, dynamic=True, mode='reduce-overhead')
            self._is_compiled = True
            log.info("Model compiled (torch.compile, dynamic=True, reduce-overhead)")
            return compiled
        except Exception as e:
            log.warning("torch.compile failed (%s); falling back to eager mode.", e)
            return model

    # ---------------------------------------------------------------------------------------------------------------------<
    def _async_d2h(self, t_gpu) -> torch.Tensor:
        """
        Start an async GPU→CPU copy into a freshly-allocated pinned buffer.
        Caller MUST torch.cuda.synchronize() (or explicitly wait on the stream)
        before reading the returned tensor as numpy.
        """
        if not self._cuda:
            return t_gpu  # already on CPU
        out_cpu = torch.empty(t_gpu.shape, dtype=t_gpu.dtype, device='cpu', pin_memory=True)
        out_cpu.copy_(t_gpu, non_blocking=True)
        return out_cpu

    # ---------------------------------------------------------------------------------------------------------------------<
    def _postprocess_numpy(self, data: np.ndarray) -> np.ndarray:
        """
        Applies post_func and the deoperator using pre-resolved callables.
        Denormalization is no longer done here — the caller has already applied
        ``self.norm_y.inverse_transform`` on the GPU prior to the D2H copy, so
        ``data`` arrives in operator-applied (e.g. sqrt) physical space.
        Returns a numpy array of shape (B, C, G) ready for xarray conversion.
        """
        if self.transform_to_2D_y:
            B, C, H, W = data.shape
            data = data.reshape(B, C, H * W)
        data = self.post_func(data, **self.post_func_kwargs)
        if self._ops_y is not None:
            for c, fn in enumerate(self._ops_y):
                if fn is not None:
                    data[:, c, :] = fn(data[:, c, :], back=True)
        return data

    # ---------------------------------------------------------------------------------------------------------------------<
    def postprocess(self, date, data, vars, member, operator=None, normalizer=None, lats=None, lons=None, template=None, func=None, kwargs=None):
        """
        Postprocesses model output: denormalization, deoperator, formatting, and conversion to xarray.
        The operator/normalizer/func/kwargs arguments are kept for API compatibility but are no longer
        used — the pre-resolved callables built in __init__ are used instead.
        """
        data = self._postprocess_numpy(data)
        date = np.datetime64(date)
        ds_pred = from_pred_to_xarray(data, date, vars, lats, lons, template, self.H_y, self.W_y)
        ds_pred = ds_pred.assign_coords({"member": member})
        return ds_pred

    # ---------------------------------------------------------------------------------------------------------------------<
    def downscale(self, model=None, return_pred=False, verbose=True,
                  batch_size=1, amp_dtype=None, compile=False):
        """
        Runs the downscaling process: preprocesses input, predicts, postprocesses, and saves or returns output.

        Loop structure: outer over date BATCHES, no inner member loop. The base
        class is for DETERMINISTIC models (DeepESD, GNN, ...), so calling the
        model `ensemble_size` times would give identical results — we instead
        compute once and broadcast across the `member` coordinate at the end.
        Stochastic subclasses (CPMGEM, ResDiff) override `downscale` and add
        an inner member loop where it actually matters.

        Parameters
        ----------
        batch_size : int
            Dates per forward pass. Larger → better GPU utilisation, more memory.
        amp_dtype  : str or None
            "bfloat16" / "float16" / None. Mixed-precision autocast for the
            forward pass. bf16 is the safe default on Ampere+; fp16 may need
            careful handling on older GPUs.
        compile    : bool
            If True, wraps the model in torch.compile(dynamic=True,
            mode="reduce-overhead"). Pays a one-shot compilation cost on the
            first batch, then much faster steady-state (CUDA-graph-backed).
        """
        if verbose:
            log.info("Starting downscaling process")
        if model is None:
            model = self.model

        # -- Inference flags --
        self._amp_dtype = self._parse_amp_dtype(amp_dtype)
        model = self._maybe_compile(model, compile)
        model.eval()

        all_dates_np = [np.datetime64(d) for d in self.target_dates]
        T = len(self.target_dates)
        n_batches = (T + batch_size - 1) // batch_size

        # -- Pipelined date loop with async D2H overlap ----------------------
        # While batch N's tensor copies GPU→CPU, batch N+1's preprocessing
        # runs on CPU and its forward starts on GPU. We only sync just before
        # we need the previous batch's bytes for postprocess.
        all_preds = []
        pending_cpu = None      # async D2H buffer from the previous batch

        for b_idx in range(n_batches):
            i = b_idx * batch_size
            batch_dates = self.target_dates[i : i + batch_size]
            if verbose:
                log.info("Batch %d/%d: %s → %s (%d dates)",
                         b_idx + 1, n_batches, batch_dates[0], batch_dates[-1], len(batch_dates))

            # -- Preprocess on CPU then one transfer to GPU --
            inp = self._stack_to_device([self._preprocess_single_date(d) for d in batch_dates])  # (B, ..., H, W)
            f   = self._stack_to_device([self._preprocess_forcing_date(d) for d in batch_dates]) # (B, Cy, ...)

            # -- GPU-side normalization (mirrors trainer) --
            if self.norm_x is not None:
                inp = self.norm_x(inp)
            if self.forcing_data is not None and self.norm_f is not None:
                f = self.norm_f(f)

            # -- Predict --
            with torch.inference_mode(), self._amp_ctx():
                if self.graph is not None:
                    p_torch = self.graphPredict(x=inp, edge_index=self.edge_index, model=model, f=f)
                else:
                    p_torch = model(inp, f)

            # -- GPU-side denormalization of the prediction back to operator space --
            # The model output is in normalized space; inverse_transform recovers
            # operator-applied values (e.g. sqrt(pr)). Operator inverse runs on
            # the CPU side inside _postprocess_numpy.
            if self.norm_y is not None:
                p_torch = self.norm_y.inverse_transform(p_torch.float())

            # -- Async D2H for this batch; flush previous --
            if pending_cpu is not None:
                if self._cuda:
                    torch.cuda.synchronize()
                all_preds.append(self._postprocess_numpy(pending_cpu.numpy()))
            pending_cpu = self._async_d2h(p_torch.float())  # cast back to float32 for numpy postprocess
            del inp, f, p_torch

        # Final flush
        if pending_cpu is not None:
            if self._cuda:
                torch.cuda.synchronize()
            all_preds.append(self._postprocess_numpy(pending_cpu.numpy()))

        # -- Build xarray ONCE; broadcast across ensemble dim --
        all_preds_np = np.concatenate(all_preds, axis=0)  # (T, C, G)
        ds = from_pred_to_xarray(
            all_preds_np, all_dates_np, self.vars_y,
            self.lats, self.lons, self.template, self.H_y, self.W_y,
            precomputed_mask=self._template_mask,
        )
        if self.ensemble_size > 1:
            # Deterministic model + ensemble_size>1: replicate across members.
            ds = ds.expand_dims(member=np.arange(self.ensemble_size))
        else:
            ds = ds.expand_dims(member=[0])

        if self.format_output:
            ds = self.formatting_func(ds, **self.formatting_kwargs)
        if return_pred:
            return ds
        log.debug("Writing prediction xarray to %s\n%s", self.output_path, ds)
        ds.to_netcdf(self.output_path)


