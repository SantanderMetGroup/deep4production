import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import zarr
import numcodecs
from torch import from_numpy

## Deep4production
from deep4production.core.pydatasets.pydataset import pydataset
from deep4production.deep.utils import load_model
from deep4production.deep.models.unet.padding import build_padder
from deep4production.deep.models.diffusion.patching import (
    build_grid_patcher,
    run_regressor_patched,
)
from deep4production.utils.log import get_logger
from deep4production.utils.distributed import barrier, is_main_process

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
    normalize_on_cpu : bool
        Set by the trainer for multi-source runs: c_low is then normalized here
        with this source's own predictor stats instead of on the GPU.
    source_name : str
        Multi-source runs only: suffixes the residuals zarr so each source gets
        its own cache (RESIDUALS_<source>_<dataset>.zarr).
    """

    def __init__(
        self,
        predictors: dict,
        predictands: dict,
        temporal_period: list,
        dataset: str = "training",
        path_regressor: str = None,
        residuals: dict = None,
        forcings: dict = None,
        load_in_memory: bool = True,
        add_pred_mean: bool = True,
        add_context_lowres: bool = True,
        standardize_residuals: bool = False,
        normalizer_info_x: dict = None,
        normalizer_info_y: dict = None,
        normalizer_info_f: dict = None,
        cache_mb: int = None,
        regressor_batch_size: int = 32,
        normalize_on_cpu: bool = False,
        source_name: str = None,
    ):
        # --- Call parent constructor (loads x/y/forcings, builds pipelines, temporal info) ---
        # The parent builds the CPU-side InputNormalizer instances this class
        # needs for the residuals precomputation: the regressor expects
        # normalized x and produces output in normalized space, and the residual
        # is computed against normalized y. For single-source runs
        # ``normalize_on_cpu`` is False and the trainer normalizes c_low on the
        # GPU; multi-source runs set it so c_low gets this source's own stats.
        #
        # NOTE: the parent forwards operator_info when resolving these, exactly
        # as trainer.py does for the non-residual runs. Without it
        # `stats_transform` is never set, so for any channel carrying an operator
        # (sqrt on pr/hurs) the affine is built from RAW-space min/max while the
        # data is in operator space. The regressor is normalized by the trainer
        # (which does pass it), so the two would end up in DIFFERENT spaces and
        # the residual r = norm_y(y) - yhat would pick up a large constant offset
        # on exactly those channels — silently, since every operator-free channel
        # is unaffected.
        super().__init__(
            predictors=predictors,
            predictands=predictands,
            forcings=forcings if forcings else {},
            temporal_period=temporal_period,
            load_in_memory=load_in_memory,
            cache_mb=cache_mb,
            normalizer_info_x=normalizer_info_x,
            normalizer_info_y=normalizer_info_y,
            normalizer_info_f=normalizer_info_f,
            normalize_on_cpu=normalize_on_cpu,
        )

        self.load_in_memory = load_in_memory
        self.add_pred_mean = add_pred_mean
        self.add_context_lowres = add_context_lowres
        self.standardize_residuals = standardize_residuals

        # --- Regressor ---
        log.info("Loading regressor model from %s", path_regressor)
        self.regressor_model, reg_meta = load_model(
            path=path_regressor, return_metadata=True
        )
        # load_model builds on CPU; run the cache forward on this rank's GPU.
        if torch.cuda.is_available():
            self.regressor_model.to(torch.device("cuda", torch.cuda.current_device()))
        # If the regressor was trained CorrDiff-patched, replicate its tiled
        # forward here so the cached residuals match inference exactly. Geometry
        # comes from the regressor checkpoint's own metadata.
        self.reg_patcher = None
        self.reg_K = 0
        rcfg = reg_meta.get("patching")
        if rcfg and rcfg.get("enabled", False):
            self.reg_patcher, self.reg_K = build_grid_patcher(
                rcfg, (self.H_y, self.W_y)
            )
            log.info(
                "Regressor is patched: tiling residual computation into %d patches.",
                self.reg_patcher.patch_num,
            )
        # Likewise, replicate the regressor's reflection padding so the cached
        # residuals are built from the same forward the downscaler will run.
        self.reg_padder = build_padder(reg_meta.get("padding"), (self.H_y, self.W_y))
        if self.reg_padder is not None:
            log.info(
                "Regressor is padded: computing residuals on a %dx%d grid.",
                self.reg_padder.padded_H, self.reg_padder.padded_W,
            )

        # --- Residuals zarr (one per source in multi-source runs) ---
        base = residuals["path"][:-5]
        if source_name:
            base = f"{base}_{source_name}"
        path_residuals_zarr = f"{base}_{dataset}.zarr"
        variables_residuals = [f"{v}_residual" for v in self.vars_y] + [
            f"{v}_normalized" for v in self.vars_y
        ]

        # Under DDP only rank 0 writes the cache; the other ranks wait and read it.
        if is_main_process():
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
        else:
            # Poll instead of barrier(): the write outlasts NCCL's collective timeout.
            while not self._residuals_zarr_valid(path_residuals_zarr):
                time.sleep(30)
        barrier()
        # The regressor is only needed to build the cache. Dropping it frees GPU
        # memory (one copy per source and split otherwise) and keeps it out of
        # the DataLoader workers.
        self.regressor_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        # --- Per-channel residual standardization (CorrDiff / EDM sigma_data=1) ---
        # The residuals zarr stores per-channel (mean, std) over the residual
        # channels (first C of 2C). When standardize_residuals is on we serve
        # (r - mean) / std so the diffused field has ~unit variance and the EDM
        # preconditioner can use sigma_data=1.0. The stats also go into the run
        # metadata so the downscaler can invert the standardization at inference.
        #
        # These are THIS store's stats. The trainer overrides them with the
        # training split's (pooled over sources, if any) via set_residual_norm,
        # so every split and source is standardized with the stats the
        # downscaler will invert.
        n_y = len(self.vars_y)
        # Unguarded store stats and their sample count, kept for pooling.
        self.residual_mean_raw = np.asarray(r_zarr[0]["mean"], dtype=np.float64)[:n_y]
        self.residual_std_raw = np.asarray(r_zarr[0]["std"], dtype=np.float64)[:n_y]
        self.residual_count = len(self.target_dates) * self.G_y
        self.set_residual_norm(self.residual_mean_raw, self.residual_std_raw)

    # -------------------------------------------------------------------------
    def set_residual_norm(self, mean, std):
        """Set the per-channel (mean, std) used to standardize the served residual."""
        n_y = len(self.vars_y)
        res_mean = np.asarray(mean, dtype=np.float32).reshape(n_y)
        res_std = np.asarray(std, dtype=np.float32).reshape(n_y)
        # Guard against degenerate (near-zero) channel std.
        res_std = np.where(res_std < 1e-8, 1.0, res_std).astype(np.float32)
        self.residual_mean = res_mean
        self.residual_std = res_std
        if self.standardize_residuals:
            # Broadcast shape: (C, 1, 1) for 2D fields, (C, 1) for flattened grids.
            bshape = (n_y, 1, 1) if self.transform_to_2D_y else (n_y, 1)
            self._res_mean_t = from_numpy(res_mean.reshape(bshape))
            self._res_std_t = from_numpy(res_std.reshape(bshape))

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
        x_in_shape = (
            self.reg_padder.padded_shape(batch_size, num_y)
            if self.reg_padder is not None
            else (batch_size, num_y, self.H_y, self.W_y)
        )
        x_in_buf = torch.zeros(x_in_shape, device=device)
        t_buf = torch.zeros(batch_size, device=device)

        self.regressor_model.eval()
        for start in range(0, T, batch_size):
            batch_dates = self.target_dates[start : start + batch_size]
            B = len(batch_dates)

            # --- CPU preprocessing for the batch ---
            x_list, y_list, f_list = [], [], []
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

                # High-res forcing (orography) on the predictand grid, fed to the
                # regressor as cond_high. Read from the predictand zarr (idx_vars_f
                # / sample_map_y), exactly as the base pydataset __getitem__.
                if self._norm_f_cpu is not None:
                    f = self.preprocess(
                        date,
                        self.data["y"],
                        self.idx_vars_f,
                        self.sample_map_y,
                        ops=self._ops_f,
                        transform_to_2D=self.transform_to_2D_y,
                        H=self.H_y,
                        W=self.W_y,
                    ).unsqueeze(0)
                    f = self._norm_f_cpu(f)
                    f_list.append(f)

            x_batch = torch.cat(x_list, dim=0).to(device)  # (B, C_x, H_x, W_x)
            y_batch = torch.cat(y_list, dim=0)  # (B, C_y, H_y, W_y)
            # cond_high for the regressor: normalized forcing or None (no forcings).
            f_batch = (
                torch.cat(f_list, dim=0).to(device) if f_list else None
            )  # (B, C_f, H_y, W_y) or None

            # --- Batched regressor forward pass ---
            with torch.no_grad():
                if self.reg_patcher is not None:
                    # Tiled forward: upsample cond_low to HR, then apply→forward→fuse.
                    x_hr = F.interpolate(
                        x_batch, size=(self.H_y, self.W_y),
                        mode="bilinear", align_corners=False,
                    )
                    reg_out = run_regressor_patched(
                        self.regressor_model, x_hr, f_batch,
                        self.reg_patcher, self.reg_K, num_y,
                    ).cpu()  # (B, C_y, H_y, W_y)
                elif self.reg_padder is not None:
                    # Padded forward, cropped back to the native predictand grid.
                    reg_out = self.reg_padder.crop(
                        self.regressor_model(
                            x=x_in_buf[:B],
                            t=t_buf[:B],
                            cond_low=self.reg_padder.apply_cond_low(x_batch),
                            cond_high=self.reg_padder.apply(f_batch),
                        )
                    ).cpu()  # (B, C_y, H_y, W_y)
                else:
                    reg_out = self.regressor_model(
                        x=x_in_buf[:B],
                        t=t_buf[:B],
                        cond_low=x_batch,
                        cond_high=f_batch,
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

        # Per-channel standardization to ~unit variance (EDM sigma_data=1). The
        # regressor mean (c_high) below is left in raw [-1,1] target space; only
        # the diffused residual is standardized. Inverted in downscaler_resdiff.
        if self.standardize_residuals:
            residual = (residual - self._res_mean_t) / self._res_std_t

        c_high = None
        if self.add_pred_mean:
            c_high = r_raw[num_vars:]
            if self.transform_to_2D_y:
                c_high = c_high.reshape(num_vars, self.H_y, self.W_y)

        # --- Low-res predictor context ---
        # preprocess() applies operator → reshape → tensor; normalization is
        # applied here for multi-source runs, otherwise by the trainer on the GPU.
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
            if self.normalize_on_cpu and self._norm_x_cpu is not None:
                c_low = self._norm_x_cpu(c_low, channel_dim=0)

        return residual, c_low, c_high

    # -------------------------------------------------------------------------
    def get_residual_norm(self):
        """
        Per-channel residual standardization stats (residual channels only),
        for the trainer to persist into run metadata so downscaler_resdiff can
        invert the standardization at inference.

        Returns
        -------
        dict with keys:
            standardize : bool   whether residuals are standardized at training
            mean        : list   per-channel residual mean (length C)
            std         : list   per-channel residual std  (length C)
        """
        return {
            "standardize": bool(self.standardize_residuals),
            "mean": [float(m) for m in self.residual_mean],
            "std": [float(s) for s in self.residual_std],
        }


# -------------------------------------------------------------------------
def pool_residual_norm(datasets):
    """
    Pool per-channel residual (mean, std) over several residual pydatasets.

    Exact for the population moments each store holds: every store's (mean, std)
    is weighted by its sample count (dates x gridpoints), so the result equals
    the stats of all residuals concatenated. Used for multi-source runs, where
    one diffusion model (and one inverse transform at inference) serves every
    source.

    Returns
    -------
    (mean, std) : two float64 arrays of length C
    """
    counts = np.array([d.residual_count for d in datasets], dtype=np.float64)
    means = np.stack([d.residual_mean_raw for d in datasets])  # (S, C)
    stds = np.stack([d.residual_std_raw for d in datasets])
    w = (counts / counts.sum())[:, None]
    mean = (w * means).sum(axis=0)
    second = (w * (stds**2 + means**2)).sum(axis=0)
    std = np.sqrt(np.maximum(second - mean**2, 0.0))
    return mean, std
