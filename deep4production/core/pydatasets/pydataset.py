import numpy as np
import xarray as xr
from torch import from_numpy
from torch.utils.data import Dataset
import torch

## Deep4production
from deep4production.utils.trans import from_pred_to_xarray
from deep4production.utils.general import get_func_from_string
from deep4production.utils.temporal import (
    get_dates_from_yaml,
    get_sample_map,
    get_pairs,
)
from deep4production.utils.zarr import open_zarr_store
from deep4production.utils.log import get_logger

log = get_logger("pydataset")


########################################################################################################
########################################################################################################
class pydataset(Dataset):
    """
    Dataset class for loading, preprocessing, and batching predictor/predictand/forcing data for deep learning.
    Purpose: Handles variable selection, normalization, operator application, temporal alignment, and batching for PyTorch models.
    Parameters:
        predictors (dict): Predictor dataset configuration.
        predictands (dict): Predictand dataset configuration.
        temporal_period (list): List of target dates.
        load_in_memory (bool): Whether to load all data into memory.
        forcings (dict, optional): Forcing dataset configuration.
    """

    def __init__(
        self,
        predictors: dict,
        predictands: dict,
        temporal_period: list,
        load_in_memory: bool = True,
        forcings={},
        cache_mb: int = None,
    ):
        # --- Parameters (X, Y) ---
        # Note: ``normalizer`` is no longer consumed by pydataset. The trainer
        # owns GPU-side InputNormalizer modules and applies them per-batch on
        # device — see deep4production.deep.preprocessing.normalizer. The recipe
        # schema is unchanged; ``cli/train.py`` extracts the normalizer dicts
        # and forwards them to the trainer directly.
        path_predictors, path_predictands = predictors["paths"], predictands["paths"]
        variables_predictors, variables_predictands, variables_forcings = (
            predictors.get("variables", None),
            predictands.get("variables", None),
            forcings.get("variables", None),
        )
        operator_predictors, operator_predictands, operator_forcings = (
            predictors.get("operator", None),
            predictands.get("operator", None),
            forcings.get("operator", None),
        )
        self.transform_to_2D_x, self.transform_to_2D_y = (
            predictors.get("transform_to_2D", False),
            predictands.get("transform_to_2D", False),
        )
        self.num_lagged_x, self.num_lagged_y = (
            predictors.get("num_lagged", 0),
            predictands.get("num_lagged", 0),
        )
        # Per-block zarr format: "d4p" (default), "anemoi", or "auto" to autodetect.
        fmt_x = predictors.get("format", "auto")
        fmt_y = predictands.get("format", "auto")
        fmt_f = forcings.get("format", "auto") if forcings else "auto"

        # --- Load metadata ---
        # cache_mb is forwarded to open_zarr_store for an LRU cache layer when
        # load_in_memory=False. When load_in_memory=True, the cache is moot
        # (data is materialized into a numpy array up front).
        self.cache_mb = cache_mb
        (
            self.x,
            self.vars_x,
            self.idx_vars_x,
            self.operator_x,
            self.H_x,
            self.W_x,
            self.G_x,
        ) = self.get_data_info(
            path_predictors,
            variables_predictors,
            operator_predictors,
            fmt=fmt_x,
            cache_mb=cache_mb,
        )
        (
            self.y,
            self.vars_y,
            self.idx_vars_y,
            self.operator_y,
            self.H_y,
            self.W_y,
            self.G_y,
        ) = self.get_data_info(
            path_predictands,
            variables_predictands,
            operator_predictands,
            fmt=fmt_y,
            cache_mb=cache_mb,
        )
        self.forcings = forcings
        if forcings:
            _, self.vars_f, self.idx_vars_f, self.operator_f, __, ___, _____ = (
                self.get_data_info(
                    path_predictands,
                    variables_forcings,
                    operator_forcings,
                    fmt=fmt_f,
                    cache_mb=cache_mb,
                )
            )
        else:
            self.vars_f = None
            self.idx_vars_f = None
            self.operator_f = None

        # --- Temporal information (intersect X and Y and get indexing info)---
        freq = self.x[0].attrs.get("frequency")
        dates_yaml = get_dates_from_yaml(temporal_period, freq=freq)
        self.sample_map_x, dates_x = get_sample_map(dates_yaml, self.x)
        self.sample_map_y, dates_y = get_sample_map(dates_yaml, self.y)
        dates = sorted(set(dates_x) & set(dates_y))
        self.pairs = get_pairs(dates=dates, freq=freq, num_lagged_x=self.num_lagged_x)
        self.target_dates = list(self.pairs.keys())
        self.num_samples = len(self.pairs)
        log.info("Number of samples: %d", self.num_samples)
        if self.num_samples == 0:
            assert "❌ There are no common dates between the predictor (X) and predictand (Y) datasets."

        # --- Load in memory? ---
        if load_in_memory:  # If dataset fits in memory, load all predictors to speed up
            x_data = [np.array(x["data"]) for x in self.x]
            y_data = [np.array(y["data"]) for y in self.y]
            self.data = {"x": x_data, "y": y_data}
            log.info("Data loaded into memory.")
        else:
            self.data = {
                "x": [x["data"] for x in self.x],
                "y": [y["data"] for y in self.y],
            }

        # --- Resolve operator callables ONCE (not per-sample) ---
        # Avoids importlib.import_module() on every ``__getitem__`` call, which
        # dominates CPU time in the dataloader. Normalization no longer happens
        # in pydataset — the trainer applies it per-batch on the GPU via
        # InputNormalizer modules built from the recipe.
        self._ops_x = self._build_operator_pipeline(self.operator_x, self.vars_x)
        self._ops_y = self._build_operator_pipeline(self.operator_y, self.vars_y)
        if forcings:
            self._ops_f = self._build_operator_pipeline(self.operator_f, self.vars_f)
        else:
            self._ops_f = None

    # -------------------------------------------------------------------------
    @staticmethod
    def _build_operator_pipeline(operator_info, vars):
        """
        Resolve per-channel operator callables once at init time.

        Returns
        -------
        ops : list[callable or None] or None
            One entry per channel (same order as ``vars``); None means no
            operator for that channel.
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

    # -------------------------------------------------------------------------
    @staticmethod
    def _resolve_normalizer_info(
        normalizer_info_recipe, vars, predictand=False, forcing=False
    ):
        """
        Turn the recipe's ``normalizer:`` block into a fully-resolved dict
        ready to feed into ``InputNormalizer``. Reads the per-variable mean /
        std / min / max from the reference Zarr's stat sub-arrays, fills in
        ``stats_transform`` if an operator was configured, and resolves the
        per-variable method name via ``default`` + per-variable overrides.

        This is a static helper used by the trainer; pydataset itself no
        longer applies normalization.
        """
        if normalizer_info_recipe is None:
            return None
        path_reference = normalizer_info_recipe["path_reference"]
        method_default = normalizer_info_recipe.get("default", None)

        zarr_file = open_zarr_store(path_reference, fmt="auto")
        kwargs_per_var = {}
        for var in vars:
            var_idx = zarr_file.attrs["variables"][var]
            kwargs_per_var[var] = {
                "mean": float(zarr_file["mean"][var_idx]),
                "std": float(zarr_file["std"][var_idx]),
                "min": float(zarr_file["min"][var_idx]),
                "max": float(zarr_file["max"][var_idx]),
            }

        normalizer_func_per_variable = {
            var: normalizer_info_recipe.get(var, method_default) for var in vars
        }

        return {
            "dataset": path_reference,
            "kwargs": kwargs_per_var,
            "normalizer_func_per_variable": normalizer_func_per_variable,
        }

    # -------------------------------------------------------------------------
    def get_data_info(
        self, path_data, variables, operator_info, fmt="auto", cache_mb=None
    ):
        """
        Loads metadata and variable info from Zarr files, plus operator setup.

        Parameters:
            path_data (list): List of Zarr file paths.
            variables (list): Variable names.
            operator_info (dict): Operator configuration.
            fmt (str): "d4p", "anemoi", or "auto" — opens stores via the
                       d4p/anemoi adapter (see utils.zarr.open_zarr_store).
            cache_mb (int or None): If set, wrap each store in a
                       zarr.LRUStoreCache of the given megabyte budget. Useful
                       for ``load_in_memory=False`` over slow filesystems.
        Returns:
            tuple: (files, vars, idx_vars, operator, H, W, G)
        """
        # --- Files ---
        files = [open_zarr_store(p, fmt=fmt, cache_mb=cache_mb) for p in path_data]
        # --- Variables ---
        log.warning(
            "Variable subsetting assumes the same variable order across all zarr files (e.g. self.x[0] and self.x[1])."
        )
        if variables is None:  # Selecting all available variables in the dataset
            vars = [var for var, idx in files[0].attrs["variables"].items()]
            idx_vars = [idx for var, idx in files[0].attrs["variables"].items()]
        else:
            vars = variables
            idx_vars = [
                files[0].attrs["variables"][var]
                for var in vars
                if var in files[0].attrs["variables"]
            ]
        # --- Operator ---
        operator = None
        if operator_info is not None:
            operator = {}
            operator_info_default = operator_info.get("default", None)
            operator["module"] = "deep4production.utils.operators"
            operator["operator_func_per_variable"] = {
                var: (
                    operator_info[var]
                    if var in operator_info
                    else operator_info_default
                )
                for var in vars
            }
            log.debug(
                "Operator for variables %s: %s",
                vars,
                operator["operator_func_per_variable"],
            )
        # --- Height and width (H and W) ---
        H, W = files[0].attrs.get("H", None), files[0].attrs.get("W", None)
        # --- Number of gridpoints (G) ---
        G = files[0].attrs.get("shape")[2]
        # --- Return ---
        return files, vars, idx_vars, operator, H, W, G

    # -------------------------------------------------------------------------
    def get_forcings_info(self):
        """
        Returns information about forcings variables, indices, and operator.
        The third element is kept as ``None`` for backwards-compatibility with
        callers expecting a 4-tuple — normalizer info no longer lives in
        pydataset (the trainer owns it now).

        Returns:
            tuple: (vars_f, idx_vars_f, None, operator_f)
        """
        return self.vars_f, self.idx_vars_f, None, self.operator_f

    # -------------------------------------------------------------------------
    def get_coords(self):
        """
        Returns latitude and longitude arrays for predictands.
        Returns:
            tuple: (lats, lons)
        """
        lats = np.array(self.y[0]["latitudes"][:], dtype=np.float32)
        lons = np.array(self.y[0]["longitudes"][:], dtype=np.float32)
        return lats, lons

    # -------------------------------------------------------------------------
    def get_spatial_dims(self):
        """
        Returns spatial dimensions (height, width) for predictors and predictands.
        Returns:
            tuple: (H_x, W_x, H_y, W_y)
        """
        return self.H_x, self.W_x, self.H_y, self.W_y

    # -------------------------------------------------------------------------
    def get_vars(self):
        """
        Returns variable names for predictors and predictands.
        Returns:
            tuple: (vars_x, vars_y)
        """
        return self.vars_x, self.vars_y

    # -------------------------------------------------------------------------
    def get_num_gridpoints(self):
        """
        Returns number of gridpoints for predictors and predictands.
        Returns:
            tuple: (G_x, G_y)
        """
        return self.G_x, self.G_y

    # -------------------------------------------------------------------------
    def get_transform2D(self):
        """
        Returns transform-to-2D flags for predictors and predictands.
        Returns:
            tuple: (transform_to_2D_x, transform_to_2D_y)
        """
        return self.transform_to_2D_x, self.transform_to_2D_y

    # -------------------------------------------------------------------------
    def get_lagged_info(self):
        """
        Returns number of lagged timesteps for predictors and predictands.
        Returns:
            tuple: (num_lagged_x, num_lagged_y)
        """
        return self.num_lagged_x, self.num_lagged_y

    # -------------------------------------------------------------------------
    def get_operator_info(self, predictands=False):
        """
        Returns operator info for predictors or predictands.
        Parameters:
            predictands (bool): If True, returns for predictands; else for predictors.
        Returns:
            dict or None: Operator info.
        """
        if predictands:
            return self.operator_y
        else:
            return self.operator_x

    # -------------------------------------------------------------------------
    def get_target_samples(self):
        """
        Returns target samples as xarray.Dataset for all dates in the dataset.
        Returns:
            xarray.Dataset: Target samples stacked along time.
        """
        target_samples = []
        # --- Loop over samples ---
        for idx in range(len(self)):
            # --- Get dates ---
            target_date = self.target_dates[idx]

            # --- Prepare target ---
            y = (
                self.preprocess(
                    target_date,
                    self.data["y"],
                    self.idx_vars_y,
                    self.sample_map_y,
                    ops=None,
                    transform_to_2D=None,
                    H=None,
                    W=None,
                )
                .unsqueeze(0)
                .cpu()
                .numpy()
            )  # Add time dimension
            # --- To xarray ---
            lats, lons = self.get_coords()
            ds = from_pred_to_xarray(
                data_pred=y,
                time_pred=np.datetime64(target_date),
                vars=self.vars_y,
                lats=lats,
                lons=lons,
            )
            target_samples.append(ds)
        # --- Stack ---
        target_samples = xr.concat(target_samples, dim="time")
        return target_samples

    # -------------------------------------------------------------------------
    def preprocess(
        self,
        date,
        data,
        idx_vars,
        sample_map,
        ops=None,
        transform_to_2D=False,
        H=None,
        W=None,
    ):
        """
        Preprocess a sample: index → operator → reshape → to-tensor.

        Normalization is no longer applied here — the trainer applies it on
        the GPU per batch via an InputNormalizer module. The operator stage
        stays on the CPU per sample because operators are non-linear and
        per-channel; their effect on min/max is propagated to the GPU
        normalizer's affine coefficients via ``stats_transform`` at build time.

        Parameters
        ----------
        date : any
            Target date key used to look up the sample in ``sample_map``.
        data : list
            Per-file data (zarr handles or in-memory ndarrays).
        idx_vars : list[int]
            Channel indices to extract from the source array.
        sample_map : dict
            Maps ``date`` → ``(file_idx, time_idx)``.
        ops : list[callable or None] or None
            Per-channel operator callables, pre-resolved by
            ``_build_operator_pipeline`` at init. ``None`` means "skip".
        transform_to_2D : bool
            Whether to reshape from ``(C, H*W)`` to ``(C, H, W)``.
        H, W : int, optional
            Target spatial dimensions when ``transform_to_2D`` is True.

        Returns
        -------
        torch.Tensor
            ``(C, G)`` or ``(C, H, W)``.
        """
        # -- Get sample --
        i, j = sample_map[date]
        x = data[i][j][idx_vars]  # (C, G)
        # --- Operator (per-channel, pre-resolved callables) ---
        if ops is not None:
            for c, fn in enumerate(ops):
                if fn is not None:
                    x[c, :] = fn(x[c, :])
        # --- Transform to 2D ---
        if transform_to_2D:
            C, G = x.shape
            x = x.reshape(C, H, W)
        # --- Convert to torch tensor ---
        return from_numpy(x)

    # -------------------------------------------------------------------------
    def __len__(self):
        """
        Returns number of samples in the dataset.
        Returns:
            int: Number of samples.
        """
        return self.num_samples

    # -------------------------------------------------------------------------
    def __getitem__(self, idx):
        """
        Returns a tuple (x, y, f) for a given sample index.
        Parameters:
            idx (int): Sample index.
        Returns:
            tuple: (x, y, f)
        """
        # --- Prepare data ---
        target_date = self.target_dates[idx]
        dates = self.pairs[target_date]
        # ---
        if len(dates) > 1:
            x = []
            for date in dates:
                x.append(
                    self.preprocess(
                        date,
                        self.data["x"],
                        self.idx_vars_x,
                        self.sample_map_x,
                        ops=self._ops_x,
                        transform_to_2D=self.transform_to_2D_x,
                        H=self.H_x,
                        W=self.W_x,
                    )
                )
            x = torch.stack(x)
        else:
            x = self.preprocess(
                target_date,
                self.data["x"],
                self.idx_vars_x,
                self.sample_map_x,
                ops=self._ops_x,
                transform_to_2D=self.transform_to_2D_x,
                H=self.H_x,
                W=self.W_x,
            )
        # ---
        y = self.preprocess(
            target_date,
            self.data["y"],
            self.idx_vars_y,
            self.sample_map_y,
            ops=self._ops_y,
            transform_to_2D=self.transform_to_2D_y,
            H=self.H_y,
            W=self.W_y,
        )
        # --- Forcings (f) ---
        if self.forcings:
            f = self.preprocess(
                target_date,
                self.data["y"],
                self.idx_vars_f,
                self.sample_map_y,
                ops=self._ops_f,
                transform_to_2D=self.transform_to_2D_y,
                H=self.H_y,
                W=self.W_y,
            )
        else:
            f = "N/A"
        # --- Return ---
        return x, y, f
