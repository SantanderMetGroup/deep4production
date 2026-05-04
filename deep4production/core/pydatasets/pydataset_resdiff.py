import os
import numpy as np
import xarray as xr
import torch
import zarr
from torch import from_numpy
## Deep4production
from deep4production.core.datasets.dataset import dataset as Dataset
from deep4production.core.pydatasets.pydataset import pydataset
from deep4production.deep.preprocessing.normalizer import InputNormalizer
from deep4production.utils.trans import from_pred_to_xarray
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
        Residuals configuration with keys 'path' and 'template'.
    load_in_memory : bool
        Whether to load all data (including residuals) into memory.
    add_pred_mean : bool
        Whether to include the deterministic prediction as high-res context.
    add_context_lowres : bool
        Whether to include low-res predictors as context.
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
        cache_mb: int = None,
    ):
        # --- Call parent constructor (loads x/y, builds pipelines, temporal info) ---
        super().__init__(
            predictors=predictors,
            predictands=predictands,
            temporal_period=temporal_period,
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
        if normalizer_info_x is not None:
            resolved = pydataset._resolve_normalizer_info(normalizer_info_x, self.vars_x, predictand=False)
            self._norm_x_cpu = InputNormalizer(resolved, self.vars_x, channel_dim=1)
        if normalizer_info_y is not None:
            resolved = pydataset._resolve_normalizer_info(normalizer_info_y, self.vars_y, predictand=True)
            self._norm_y_cpu = InputNormalizer(resolved, self.vars_y, channel_dim=1)

        # --- Regressor ---
        log.info("Loading regressor model from %s", path_regressor)
        self.regressor_model = load_model(path=path_regressor)

        # --- Residuals zarr ---
        path_residuals_zarr = f"{residuals['path'][:-5]}_{dataset}.zarr"
        variables_residuals = (
            [f"{v}_residual" for v in self.vars_y]
            + [f"{v}_normalized" for v in self.vars_y]
        )

        if not os.path.exists(path_residuals_zarr):
            log.info("Producing residuals (netcdf and zarr files)")
            freq = self.x[0].attrs.get("frequency")
            template = xr.open_dataset(residuals["template"])
            for idx, date in enumerate(self.target_dates):
                log.debug("Generating residual for date %s", date)
                self._forward_pass_regressor(f"./aux_residuals_{idx}.nc", date=date, template=template)
            template.close()
            Dataset(
                date_init=self.target_dates[0],
                date_end=self.target_dates[-1],
                freq=freq,
                data={
                    "paths": [f"./aux_residuals_{idx}.nc" for idx in range(len(self.target_dates))],
                    "vars": variables_residuals,
                },
            ).to_disk(path_residuals_zarr)
            for idx in range(len(self.target_dates)):
                os.remove(f"./aux_residuals_{idx}.nc")
        else:
            log.info("Residuals zarr already available at %s, skipping computation.", path_residuals_zarr)

        # --- Sample map for residuals: date -> [zarr_file_idx, time_idx] ---
        # Residuals zarr is written in target_dates order, so the time index is
        # simply the position of each date in target_dates.
        self.sample_map_r = {date: [0, idx] for idx, date in enumerate(self.target_dates)}

        # --- Open residuals zarr and optionally load into memory ---
        r_zarr = [zarr.open(path_residuals_zarr, mode="r")]
        if load_in_memory:
            log.info("Loading residuals into memory.")
            self.data["r"] = [np.array(r["data"]) for r in r_zarr]
        else:
            self.data["r"] = [r["data"] for r in r_zarr]

    # -------------------------------------------------------------------------
    def _forward_pass_regressor(self, path, date, template):
        """
        Run the regressor for one date, compute the residual, and save to NetCDF.

        Uses the parent's preprocess() so operator → normalizer → reshape → tensor
        is applied identically to what the dataloader sees at training time.

        Parameters
        ----------
        path : str
            Output NetCDF file path.
        date : str
            Target date (YYYY-MM-DD).
        template : xr.Dataset
            Coordinate template for from_pred_to_xarray.
        """
        x = self.preprocess(
            date, self.data["x"], self.idx_vars_x, self.sample_map_x,
            ops=self._ops_x,
            transform_to_2D=self.transform_to_2D_x, H=self.H_x, W=self.W_x,
        ).unsqueeze(0)  # (1, C_x, H_x, W_x) — un-normalized
        if self._norm_x_cpu is not None:
            x = self._norm_x_cpu(x)  # CPU normalize before regressor

        y = self.preprocess(
            date, self.data["y"], self.idx_vars_y, self.sample_map_y,
            ops=self._ops_y,
            transform_to_2D=self.transform_to_2D_y, H=self.H_y, W=self.W_y,
        ).unsqueeze(0)  # (1, C_y, H_y, W_y) — un-normalized
        if self._norm_y_cpu is not None:
            y = self._norm_y_cpu(y)  # CPU normalize so residual is in normalized space

        # The regressor is a SongUNet trained deterministically (trainer_song_unet_det):
        # noisy-input slot receives zeros and t is fixed at 0; it conditions via cond_low.
        device = next(self.regressor_model.parameters()).device
        num_y = len(self.vars_y)
        x_in = torch.zeros(1, num_y, self.H_y, self.W_y, device=device)
        t    = torch.zeros(1, device=device)

        with torch.no_grad():
            regressor_output = self.regressor_model(x=x_in, t=t, cond_low=x.to(device))

        residual_np = np.array(y) - np.array(regressor_output.detach().cpu())
        pred_np = np.array(regressor_output.detach().cpu())

        lats, lons = self.get_coords()
        # Pass all channels at once — from_pred_to_xarray loops over vars internally.
        # residual_np / pred_np are (1, C, H, W) so ndim==4 and H/W are inferred correctly.
        residual_ds = from_pred_to_xarray(
            residual_np[0:1], date,
            [f"{var}_residual" for var in self.vars_y],
            lats, lons, template=template,
        )
        pred_ds = from_pred_to_xarray(
            pred_np[0:1], date,
            [f"{var}_normalized" for var in self.vars_y],
            lats, lons, template=template,
        )
        xr.merge([residual_ds, pred_ds]).to_netcdf(path)

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

        c_high = None
        if self.add_pred_mean:
            c_high = r_raw[num_vars:]
            if self.transform_to_2D_y:
                c_high = c_high.reshape(num_vars, self.H_y, self.W_y)

        # --- Low-res predictor context ---
        # preprocess() applies operator → reshape → tensor; normalization is
        # applied later by the trainer on the GPU.
        c_low = None
        if self.add_context_lowres:
            c_low = self.preprocess(
                target_date, self.data["x"], self.idx_vars_x, self.sample_map_x,
                ops=self._ops_x,
                transform_to_2D=self.transform_to_2D_x, H=self.H_x, W=self.W_x,
            )

        return residual, c_low, c_high
