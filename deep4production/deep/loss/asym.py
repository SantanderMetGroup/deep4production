"""
Asymmetric loss function for precipitation downscaling.

Authors:
    Jose González-Abad
    Alfonso Hernanz
    Jorge Baño-Medina
"""

import os
import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np
import xarray as xr
import scipy.stats
from typing import Literal
from deep4production.utils.zarr import open_zarr_store
from deep4production.utils.log import get_logger

log = get_logger("deep.loss.asym")


### -------------------------------------------------------------------------------- ###
### -------------------- Asymmetric Loss ------------------------------------------- ###
class Asym(nn.Module):

    """
    Asymmetric loss function for precipitation downscaling.
    Purpose: Computes asymmetric loss using fitted gamma distributions.
    Parameters:
        ref_path (str): Reference data path.
        var (str): Target variable.
        ignore_nans (bool): Ignore NaNs in target domain.
        asym_path (str): Path to save/load gamma parameters.
        type (str): Fitting type ('per_year' or 'full').
        asym_weight (float): Weight for asymmetric term.
        cdf_pow (float): Power for CDF term.
        threshold (float): Threshold for wet days.
        appendix (str): File appendix.
    """

    def __init__(self, ref_path: str, var: str,
                 ignore_nans: bool, asym_path: str,
                 type: Literal["per_year", "full"] = "full",
                 asym_weight: float = 1.0, cdf_pow: float = 2.0, threshold: float=1.0,
                 appendix: str = None, *args, **kwargs) -> None:
        super(Asym, self).__init__()

        # --- Ensure that asym_weight and cdf_pow are numeric values ---
        if not isinstance(asym_weight, (int, float)):
            raise ValueError("'asym_weight' must be a numeric value.")
        if not isinstance(cdf_pow, (int, float)):
            raise ValueError("'cdf_pow' must be a numeric value.")

        # --- Convert to float if needed and check positiveness ---
        asym_weight = float(asym_weight)
        cdf_pow = float(cdf_pow)
        if asym_weight < 0 or cdf_pow < 0:
            raise ValueError("'asym_weight' and 'cdf_pow' must be positive.")

        # --- Device ---
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')

        # --- Store as SELF parameters ---
        self.ignore_nans = ignore_nans
        self.asym_weight = asym_weight
        self.cdf_pow = cdf_pow
        self.threshold = threshold

        # --- Saving paths ---
        shape_file_name = 'shape.npy'
        scale_file_name = 'scale.npy'
        loc_file_name = 'loc.npy'
        if appendix:
            shape_file_name = f'shape_{appendix}.npy'
            scale_file_name = f'scale_{appendix}.npy'
            loc_file_name = f'loc_{appendix}.npy'
        self.shape_path = f'{asym_path}/{shape_file_name}'
        self.scale_path = f'{asym_path}/{scale_file_name}'
        self.loc_path = f'{asym_path}/{loc_file_name}'

        # --- Get shape, scale and loc ---
        if not self.parameters_exist():
            if ref_path[-5:] == ".zarr":
                z = open_zarr_store(ref_path, fmt="auto")
                idx = z.attrs["variables"][var]
                data = np.array(z["data"][:, idx, :]).squeeze()  # From (B, C, GP) to (B, GP)
                dates = z["dates"][:].astype('datetime64[Y]').astype(str).tolist()
            elif ref_path[-3:] == ".nc":
                z = xr.open_dataset(ref_path)[var]
                data = z.values
                dates = [str(date)[:4] for date in z.time.values]
                data = data.reshape(len(dates), -1)
            log.info("ASYM loss: estimating Gamma parameters")
            self.compute_parameters(data, dates, type=type)
            log.info("ASYM loss: Gamma parameters saved at %s", asym_path)

        # --- Prepare shape, scale and loc ---
        shape, scale, loc = self.load_parameters()
        log.info("ASYM loss: Gamma parameters loaded from %s", asym_path)
        self.shape, self.scale, self.loc = self.prepare_parameters(shape, scale, loc)

    def parameters_exist(self):
        """
        Checks if gamma distribution parameters exist.
        Returns:
            bool: True if parameters exist.
        """

        shape_exist = os.path.exists(self.shape_path)
        scale_exist = os.path.exists(self.scale_path)
        loc_exist = os.path.exists(self.loc_path)
        return (shape_exist and scale_exist and loc_exist)

    def load_parameters(self):
        """
        Loads gamma distribution parameters from files.
        Returns:
            tuple: (shape, scale, loc) arrays.
        """

        shape = np.load(self.shape_path)
        scale = np.load(self.scale_path)
        loc = np.load(self.loc_path)
        return shape, scale, loc

    def _compute_gamma_parameters(self, x: np.ndarray, threshold: float=1.0) -> tuple:

        """
        Fits gamma distribution to wet days in 1D array.
        Parameters:
            x (np.ndarray): Precipitation values.
            threshold (float): Wet day threshold.
        Returns:
            tuple: (shape, loc, scale) parameters.
        """

        # If nan return nan
        if np.sum(np.isnan(x)) == len(x):
            return np.nan, np.nan, np.nan
        else:
            x = x[~np.isnan(x)] # Remove nans
            x = x[x >= threshold] # Filter wet days
            try: # Compute dist.
                fit_shape, fit_loc, fit_scale = scipy.stats.gamma.fit(x)
            except: # If its not possible return nan
                fit_shape, fit_loc, fit_scale = np.nan, np.nan, np.nan
            return fit_shape, fit_loc, fit_scale

    def compute_parameters(self, data, dates=None, type="full"):
        """
        Computes gamma parameters for each spatial gridpoint.
        Parameters:
            data (np.ndarray): Input data array.
            dates (list): List of dates.
            type (str): Fitting type.
        Returns:
            None
        """

        # --- Fit a Gamma distribution ---
        if type == "per_year":
            years = np.unique(dates)
            gamma_params = []
            for year in years:
                idx = [i for i, y in enumerate(dates) if y == year]  # list of indices
                params_year = np.apply_along_axis(self._compute_gamma_parameters, axis=0, arr=data[idx,:], threshold=self.threshold)
                gamma_params.append(params_year)
            gamma_params = np.nanmean(np.stack(gamma_params), axis=0)
        elif type == "full":
            gamma_params = np.apply_along_axis(self._compute_gamma_parameters, axis=0, arr=data, threshold=self.threshold)

        # --- Subset Gamma parameters ---
        shape = gamma_params[0, :]
        scale = gamma_params[2, :]
        loc = gamma_params[1, :]

        # --- Save the parameters in the asym_path ---
        np.save(file=self.shape_path, arr=shape)
        np.save(file=self.scale_path, arr=scale)
        np.save(file=self.loc_path, arr=loc)

    def prepare_parameters(self, shape, scale, loc):
        """
        Converts parameters to torch tensors and handles NaNs.
        Parameters:
            shape, scale, loc: Gamma parameters.
        Returns:
            tuple: (shape, scale, loc) tensors.
        """
        # --- Convert to torch tensor ---
        shape = torch.tensor(shape).to(self.device)
        scale = torch.tensor(scale).to(self.device)
        loc = torch.tensor(loc).to(self.device)

        # --- Cases where Gamma estimated NaNs for shape, scale and loc parameters ---
        epsilon = 0.0000001
        if torch.isnan(shape).any():
            shape[torch.isnan(shape)] = epsilon
        if torch.isnan(scale).any():
            scale[torch.isnan(scale)] = epsilon
        if torch.isnan(loc).any():
            loc[torch.isnan(loc)] = 0

        # --- Return ---
        return shape, scale, loc

    def compute_cdf(self, data: torch.Tensor) -> torch.Tensor:
        """
        Computes CDF for input data using fitted gamma parameters.
        Parameters:
            data (torch.Tensor): Input data.
        Returns:
            torch.Tensor: CDF values.
        """

        # Compute cdfs for Torch
        if isinstance(data, torch.Tensor):
            data = data - self.loc # For scipy, loc corresponds to the mean
            data[data < 0] = 0 # Remove the negative values, which are automatically handled by scipy
            m = td.Gamma(concentration=self.shape,
                         rate=1/self.scale,
                         validate_args=False) # Deactivates the validation of the paremeters (e.g., support)
                                              # In this way the cdf method handles nans
            cdfs = m.cdf(data)

        # Compute cdfs for Numpy
        elif isinstance(data, np.ndarray):
            cdfs = np.empty_like(data)
            cdfs = scipy.stats.gamma.cdf(data,
                                         a=self.shape, scale=self.scale, loc=self.loc)

        else:
            raise ValueError('Unsupported type for the data argument.')

        return cdfs

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Computes asymmetric loss between target and output.
        Parameters:
            target (torch.Tensor): Target data.
            output (torch.Tensor): Model output.
        Returns:
            torch.Tensor: Loss value.
        """

        # --- Only univariate cases ---
        assert target.shape[1] == 1, f"Expected univariate target (C=1), got {target.shape[1]}"

        # --- Handle both spatial (H, W) and flattened (GP) shapes ---
        if target.ndim > 3: # stack spatial dimensions
            B, C, H, W = target.shape
            target = target.reshape(B, -1) # From shape: (B, C, H, W) to (B, [C=1]*H*W)
        if output.ndim > 3: # stack spatial dimensions
            B, C, H, W = output.shape
            output = output.reshape(B, -1) # From shape: (B, C, H, W) to (B, [C=1]*H*W)

        # --- Compute CDF ---
        cdfs = self.compute_cdf(data=target)
        cdfs = torch.nan_to_num(cdfs, nan=0.0)

        # ---  Remove Nans if present ---
        if self.ignore_nans:
            nans_idx = torch.isnan(target)
            output = output[~nans_idx]
            target = target[~nans_idx]
            cdfs = cdfs[~nans_idx]

        # --- Compute loss and return ---
        loss_mae = torch.mean(torch.abs(target - output))
        loss_asym = torch.mean((cdfs ** self.cdf_pow) * torch.max(torch.tensor(0.0), target - output))
        loss = loss_mae + self.asym_weight * loss_asym
        return loss
