import os
import numpy as np
import torch
import zarr
import numcodecs
from torch import from_numpy

## Deep4production
from deep4production.core.pydatasets.pydataset import pydataset
from deep4production.deep.preprocessing.normalizer import InputNormalizer
from deep4production.deep.utils import load_model
from deep4production.utils.log import get_logger

log = get_logger("pydataset.resdiff")


########################################################################################################
class pydataset_custom(pydataset):
    """
    Dataset class for residual-based diffusion models (e.g. CorrDiff-style).
    Extends pydataset to compute and store regression residuals, and expose
    (residual, c_low, c_high) tuples for diffusion model training.

    Parameters
    ----------
    predictors : dict
        Predictor dataset configuration.
    predictands : dict
        Predictand dataset configuration.
    temporal_period : list
        List of target years.
    dataset : str
        'training' or 'validation' — used to name the residuals zarr file.
    path_regressor : str
        Path to the pre-trained regressor (deterministic) model.
    residuals : dict
        Residuals configuration with key 'path'.
    load_in_memory : bool
        Whether to load all data (including residuals) into memory.
    add_pred_mean : bool
        Whether to include the deterministic prediction as high-res context.
    add_context_lowres : bool
        Whether to include low-res predictors as context.
    regressor_batch_size : int
        Number of dates processed per regressor forward pass. Larger values
        increase GPU utilisation during residuals precomputation.
    """

    def __init__(
        self,
        predictors: dict,
        predictands: dict,
        temporal_period: list,
        dataset: str = "training",
        path_regressor: str = None,
        residuals: dict = None,
        load_in_memory: bool = True,
        add_pred_mean: bool = True,
        add_context_lowres: bool = True,
        normalizer_info_x: dict = None,
        normalizer_info_y: dict = None,
        normalizer_info_f: dict = None,
        forcings: dict = None,
        cache_mb: int = None,
        regressor_batch_size: int = 32,
    ):
        # --- Call parent constructor (loads x/y, builds pipelines, temporal info) ---
        # Forwarding `forcings` lets the parent set up self.vars_f / idx_vars_f /
        # _ops_f so high-res forcings (e.g. orography) can be served as extra
        # cond_high channels alongside the regressor mean. Empty/None → no
        # forcings, identical to the previous behavior.
        super().__init__(
            predictors=predictors,
            predictands=predictands,
            temporal_period=temporal_period,
            forcings=forcings or {},
            load_in_memory=load_in_memory,
            cache_mb=cache_mb,
        )

        self.load_in_memory = load_in_memory
        self.add_pred_mean = add_pred_mean
        self.add_context_lowres = add_context_lowres

        # --- Local InputNormalizer instances for the residuals precomputation ---
        # The regressor expects normalized x and produces output in normalized
        # space; the residual is computed against normalized y. We build local
        # CPU-side InputNormalizer instances here from the recipe dicts. The
        # trainer holds its own copies on the GPU for the actual training loop.
        self._norm_x_cpu = None
        self._norm_y_cpu = None
        self._norm_f_cpu = None
        if normalizer_info_x is not None:
            resolved = pydataset._resolve_normalizer_info(
                normalizer_info_x, self.vars_x, predictand=False
            )
            self._norm_x_cpu = InputNormalizer(resolved, self.vars_x, channel_dim=1)
        if normalizer_info_y is not None:
            resolved = pydataset._resolve_normalizer_info(
                normalizer_info_y, self.vars_y, predictand=True
            )
            self._norm_y_cpu = InputNormalizer(resolved, self.vars_y, channel_dim=1)
        # Forcings live in the cond_high stream and are concatenated onto the
        # regressor mean in __getitem__, so — unlike the standard pydataset, which
        # normalizes f GPU-side in the trainer — they must be normalized here on
        # CPU before that concat (consistent with the regressor mean, which is
        # already stored normalized in the residuals zarr).
        if self.forcings and normalizer_info_f is not None:
            resolved = pydataset._resolve_normalizer_info(
                normalizer_info_f, self.vars_f, forcing=True
            )
            self._norm_f_cpu = InputNormalizer(resolved, self.vars_f, channel_dim=1)

        # --- Regressor ---
        log.info("Loading regressor model from %s", path_regressor)
        self.regressor_model = load_model(path=path_regressor)

        # --- Residuals zarr ---
        path_residuals_zarr = f"{residuals['path'][:-5]}_{dataset}.zarr"
        variables_residuals = [f"{v}_residual" for v in self.vars_y] + [
            f"{v}_normalized" for v in self.vars_y
        ]

        if not self._residuals_zarr_valid(path_residuals_zarr):
            if os.path.exists(path_residuals_zarr):
                log.warning(
                    "Residuals zarr at %s is invalid or incomplete (stale from a "
                    "previous failed run). Recomputing.",
                    path_residuals_zarr,
                )
            self._write_residuals_zarr(
                path_residuals_zarr,
                variables_residuals,
                batch_size=regressor_batch_size,
            )
        else:
            log.info(
                "Residuals zarr already available at %s, skipping computation.",
                path_residuals_zarr,
            )

        # --- Sample map for residuals: date -> [zarr_file_idx, time_idx] ---
        # Residuals zarr is written in target_dates order, so the time index is
        # simply the position of each date in target_dates.
        self.sample_map_r = {
            date: [0, idx] for idx, date in enumerate(self.target_dates)
        }

        # --- Open residuals zarr and optionally load into memory ---
        r_zarr = [zarr.open(path_residuals_zarr, mode="r")]
        if load_in_memory:
            log.info("Loading residuals into memory.")
            self.data["r"] = [np.array(r["data"]) for r in r_zarr]
        else:
            self.data["r"] = [r["data"] for r in r_zarr]

    # -------------------------------------------------------------------------
    @staticmethod
    def _residuals_zarr_valid(path: str) -> bool:
        """Return True only if path is a complete d4p residuals zarr group."""
        if not os.path.exists(path):
            return False
        try:
            store = zarr.open(path, mode="r")
            return (
                isinstance(store, zarr.hierarchy.Group)
                and "data" in store
                and "mean" in store  # written last by _write_residuals_zarr
            )
        except Exception:
            return False

    # -------------------------------------------------------------------------
    def _write_residuals_zarr(
        self, path: str, variables_residuals: list, batch_size: int = 32
    ):
        """
        Run the frozen regressor over all target dates in batches and write
        (residual, normalized_prediction) directly to a d4p v2 zarr store.
        """
        blk = numcodecs.Blosc(cname="zstd", clevel=5)
        device = next(self.regressor_model.parameters()).device
        num_y = len(self.vars_y)
        T = len(self.target_dates)
        G = self.G_y  # total gridpoints (H_y * W_y for regular grids)

        log.info(
            "Writing residuals zarr at %s (%d dates, batch_size=%d).",
            path,
            T,
            batch_size,
        )

        # --- Create zarr group ---
        zarr_group = zarr.open_group(path, mode="w")
        data_arr = zarr_group.create_dataset(
            "data",
            shape=(T, 2 * num_y, G),
            chunks=(1, 2 * num_y, G),
            dtype="float32",
            compressor=blk,
            fill_value=np.nan,
        )

        # --- Coordinates from the predictand zarr ---
        lats = self.y[0]["latitudes"][:].astype(np.float32)
        lons = self.y[0]["longitudes"][:].astype(np.float32)
        dates_dt = np.array(self.target_dates, dtype="datetime64[ns]").astype(
            "datetime64[s]"
        )
        zarr_group.create_dataset(
            "dates", data=dates_dt, chunks=(T,), dtype="datetime64[s]", compressor=blk
        )
        zarr_group.create_dataset(
            "latitudes", data=lats, chunks=(len(lats),), dtype="float32", compressor=blk
        )
        zarr_group.create_dataset(
            "longitudes",
            data=lons,
            chunks=(len(lons),),
            dtype="float32",
            compressor=blk,
        )

        # --- Group attributes (d4p v2 format) ---
        freq = self.x[0].attrs.get("frequency")
        zarr_group.attrs["format_version"] = 2
        zarr_group.attrs["date_init_yaml"] = str(self.target_dates[0])
        zarr_group.attrs["date_end_yaml"] = str(self.target_dates[-1])
        zarr_group.attrs["num_samples"] = T
        zarr_group.attrs["num_samples_yaml"] = T
        zarr_group.attrs["frequency"] = freq
        zarr_group.attrs["variables"] = {
            v: i for i, v in enumerate(variables_residuals)
        }
        zarr_group.attrs["name_dims"] = ["time", "variable", "gridpoint"]
        zarr_group.attrs["shape"] = [T, 2 * num_y, G]
        zarr_group.attrs["is_regular"] = True
        zarr_group.attrs["H"] = self.H_y
        zarr_group.attrs["W"] = self.W_y
        zarr_group.attrs["units"] = {v: "N/A" for v in variables_residuals}
        zarr_group.attrs["variables_metadata"] = {
            v: {"computed_forcing": False, "constant_in_time": False}
            for v in variables_residuals
        }
        zarr_group.attrs["constant_fields"] = []
        zarr_group.attrs["idx_fixed_nan"] = {v: [] for v in variables_residuals}
        zarr_group.attrs["idx_dynamic_nan"] = {v: [] for v in variables_residuals}

        # --- Running stats (computed on the fly — avoids a second full data pass) ---
        n_ch = 2 * num_y
        _sum = np.zeros(n_ch, dtype=np.float64)
        _sum2 = np.zeros(n_ch, dtype=np.float64)
        _min = np.full(n_ch, np.inf, dtype=np.float64)
        _max = np.full(n_ch, -np.inf, dtype=np.float64)

        # Pre-allocate fixed-size zero tensors for the regressor's noisy-input
        # slot and time embedding (always zero for the deterministic regressor).
        x_in_buf = torch.zeros(batch_size, num_y, self.H_y, self.W_y, device=device)
        t_buf = torch.zeros(batch_size, device=device)

        self.regressor_model.eval()
        for start in range(0, T, batch_size):
            batch_dates = self.target_dates[start : start + batch_size]
            B = len(batch_dates)

            # --- CPU preprocessing for the batch ---
            x_list, y_list = [], []
            for date in batch_dates:
                x = self.preprocess(
                    date,
                    self.data["x"],
                    self.idx_vars_x,
                    self.sample_map_x,
                    ops=self._ops_x,
                    transform_to_2D=self.transform_to_2D_x,
                    H=self.H_x,
                    W=self.W_x,
                ).unsqueeze(0)
                if self._norm_x_cpu is not None:
                    x = self._norm_x_cpu(x)
                x_list.append(x)

                y = self.preprocess(
                    date,
                    self.data["y"],
                    self.idx_vars_y,
                    self.sample_map_y,
                    ops=self._ops_y,
                    transform_to_2D=self.transform_to_2D_y,
                    H=self.H_y,
                    W=self.W_y,
                ).unsqueeze(0)
                if self._norm_y_cpu is not None:
                    y = self._norm_y_cpu(y)
                y_list.append(y)

            x_batch = torch.cat(x_list, dim=0).to(device)  # (B, C_x, H_x, W_x)
            y_batch = torch.cat(y_list, dim=0)  # (B, C_y, H_y, W_y)

            # --- Batched regressor forward pass ---
            with torch.no_grad():
                reg_out = self.regressor_model(
                    x=x_in_buf[:B],
                    t=t_buf[:B],
                    cond_low=x_batch,
                ).cpu()  # (B, C_y, H_y, W_y)

            # Flatten spatial dims: (B, C_y, G)
            residuals_np = (y_batch - reg_out).numpy().reshape(B, num_y, G)
            preds_np = reg_out.numpy().reshape(B, num_y, G)

            # Write whole batch in one zarr call and accumulate stats
            batch_chunk = np.concatenate([residuals_np, preds_np], axis=1)  # (B, 2C, G)
            data_arr[start : start + B] = batch_chunk
            _sum += batch_chunk.sum(axis=(0, 2))
            _sum2 += (batch_chunk**2).sum(axis=(0, 2))
            _min = np.minimum(_min, batch_chunk.min(axis=(0, 2)))
            _max = np.maximum(_max, batch_chunk.max(axis=(0, 2)))

            log.info("Residuals: %d / %d dates.", min(start + batch_size, T), T)

        # --- Write per-channel statistics ---
        N = T * G
        mean_arr = (_sum / N).astype(np.float32)
        std_arr = np.sqrt(np.maximum(_sum2 / N - (_sum / N) ** 2, 0)).astype(np.float32)
        min_arr = _min.astype(np.float32)
        max_arr = _max.astype(np.float32)
        zarr_group.create_dataset(
            "mean", data=mean_arr, chunks=(n_ch,), dtype="float32", compressor=blk
        )
        zarr_group.create_dataset(
            "std", data=std_arr, chunks=(n_ch,), dtype="float32", compressor=blk
        )
        zarr_group.create_dataset(
            "min", data=min_arr, chunks=(n_ch,), dtype="float32", compressor=blk
        )
        zarr_group.create_dataset(
            "max", data=max_arr, chunks=(n_ch,), dtype="float32", compressor=blk
        )

        log.info("Saved residuals store at %s", path)

    # -------------------------------------------------------------------------
    def __getitem__(self, idx):
        """
        Returns (residual, c_low, c_high) for a given sample index.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        residual : torch.Tensor  (C, H, W) or (C, G)
            Regression residual for the target date.
        c_low : torch.Tensor or None  (C_x, H_x, W_x) or (C_x, G_x)
            Low-res predictor context (None if add_context_lowres=False).
        c_high : torch.Tensor or None  (C, H, W) or (C, G)
            Deterministic prediction context (None if add_pred_mean=False).
        """
        target_date = self.target_dates[idx]
        num_vars = len(self.vars_y)

        # --- Residuals and deterministic prediction from residuals zarr ---
        i, j = self.sample_map_r[target_date]
        r_raw = from_numpy(self.data["r"][i][j].astype(np.float32))  # (2*C, G)

        residual = r_raw[:num_vars]
        if self.transform_to_2D_y:
            residual = residual.reshape(num_vars, self.H_y, self.W_y)

        # --- High-res conditioning (cond_high) ---------------------------------
        # cond_high = [ŷ_det, f]: the regressor mean (if add_pred_mean) followed
        # by any high-res forcings (if configured). The order — regressor mean
        # first, forcing second — is fixed and MUST match the downscaler's concat.
        c_high_parts = []
        if self.add_pred_mean:
            c_high_det = r_raw[num_vars:]
            if self.transform_to_2D_y:
                c_high_det = c_high_det.reshape(num_vars, self.H_y, self.W_y)
            c_high_parts.append(c_high_det)

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
            if self._norm_f_cpu is not None:
                f = self._norm_f_cpu(f.unsqueeze(0)).squeeze(0)
            c_high_parts.append(f)

        c_high = torch.cat(c_high_parts, dim=0) if c_high_parts else None

        # --- Low-res predictor context ---
        # preprocess() applies operator → reshape → tensor; normalization is
        # applied later by the trainer on the GPU.
        c_low = None
        if self.add_context_lowres:
            c_low = self.preprocess(
                target_date,
                self.data["x"],
                self.idx_vars_x,
                self.sample_map_x,
                ops=self._ops_x,
                transform_to_2D=self.transform_to_2D_x,
                H=self.H_x,
                W=self.W_x,
            )

        return residual, c_low, c_high
