"""
This module contains loss functions for training deep learning
downscaling models.

Authors:
    Jose González-Abad
    Alfonso Hernanz
    Jorge Baño-Medina
"""

import os
import zarr
import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
import numpy as np
import xarray as xr
import scipy.stats
from typing import Union
from typing import Literal

### ---------------------------------------------------------------------------------------- ###
### -------------------- Mean Absolute Error Loss ------------------------------------------ ###
class MaeLoss(nn.Module):

    """
    Standard Mean Absolute Error (MAE). It is possible to compute
    this metric over a target dataset with nans.

    Parameters
    ----------
    ignore_nans : bool
        Whether to allow the loss function to ignore nans in the
        target domain.

    target : torch.Tensor
        Target/ground-truth data

    output : torch.Tensor
        Predicted data (model's output)
    """

    def __init__(self, ignore_nans: bool) -> None:
        super(MaeLoss, self).__init__()
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:

        """
        Computes MAE loss between target and output.
        Parameters:
            target (torch.Tensor): Target data.
            output (torch.Tensor): Model output.
        Returns:
            torch.Tensor: Loss value.
        """

        # --- Handle both spatial (H, W) and flattened (GP) shapes ---
        if target.ndim > 3: # stack spatial dimensions
            B, C, H, W = target.shape
            target = target.reshape(B, C, -1) # From shape: (B, C, H, W) to (B, C, H*W)
        if output.ndim > 3: # stack spatial dimensions
            B, C, H, W = output.shape
            output = output.reshape(B, C, -1) # From shape: (B, C, H, W) to (B, C, H*W)
        
        # --- Remove Nans if present ---
        if self.ignore_nans:
            nans_idx = torch.isnan(target)
            output = output[~nans_idx]
            target = target[~nans_idx]

        # --- Compute loss and return ---
        loss = torch.mean(torch.abs(target - output))
        return loss


### ---------------------------------------------------------------------------------------- ###
### -------------------- Mean Squared Error Loss ------------------------------------------- ###
class MseLoss(nn.Module):

    """
    Standard Mean Square Error (MSE) loss.
    Purpose: Computes MSE between target and output, optionally ignoring NaNs.
    Parameters:
        ignore_nans (bool): Ignore NaNs in target domain.
    """

    def __init__(self, ignore_nans: bool) -> None:
        super(MseLoss, self).__init__()
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Computes MSE loss between target and output.
        Parameters:
            target (torch.Tensor): Target data.
            output (torch.Tensor): Model output.
        Returns:
            torch.Tensor: Loss value.
        """
        
        # --- Handle both spatial (H, W) and flattened (GP) shapes ---
        if target.ndim > 3: # stack spatial dimensions
            B, C, H, W = target.shape
            target = target.reshape(B, C, -1) # From shape: (B, C, H, W) to (B, C, H*W)
        if output.ndim > 3: # stack spatial dimensions
            B, C, H, W = output.shape
            output = output.reshape(B, C, -1) # From shape: (B, C, H, W) to (B, C, H*W)

        # --- Remove Nans if present ---
        if self.ignore_nans:
            nans_idx = torch.isnan(target)
            output = output[~nans_idx]
            target = target[~nans_idx]

        # --- Compute loss and return ---
        loss = torch.mean((target - output) ** 2)
        return loss

### ------------------------------------------------------------------------------------------------- ###
### -------------------- Neg Log-likelihood Gaussian Loss ------------------------------------------- ###
class NLLGaussianLoss(nn.Module):

    """
    Negative Log-Likelihood Gaussian loss.
    Purpose: Computes NLL for Gaussian distribution, optionally ignoring NaNs.
    Parameters:
        ignore_nans (bool): Ignore NaNs in target domain.
    """

    def __init__(self, ignore_nans: bool) -> None:
        super(NLLGaussianLoss, self).__init__()
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Computes NLL Gaussian loss between target and output.
        Parameters:
            target (torch.Tensor): Target data.
            output (torch.Tensor): Model output (mean, log_var).
        Returns:
            torch.Tensor: Loss value.
        """

        # --- Ensure same shape compatibility ---
        assert output.shape[2] == 2, f"Expected P=2 (mean, log_var), got {output.shape[2]}"
        
        # --- Handle both spatial (H, W) and flattened (GP) shapes ---
        if target.ndim > 3:
            B, C, H, W = target.shape
            target = target.reshape(B, C, -1)
        else:
            B, C, GP = target.shape
            
        if output.ndim > 4:
            B, C, P, H, W = output.shape
            output = output.reshape(B, C, P, -1)
        else:
            B, C, P, GP = output.shape

        # --- Split mean and log-variance --- #
        mean = output[:, :, 0, :]      # (B, C, GP)
        log_var = output[:, :, 1, :]   # (B, C, GP)
        precision = torch.exp(-log_var)


        # --- Remove Nans if present ---
        if self.ignore_nans:
            nans_idx = torch.isnan(target)
            mean = mean[~nans_idx]
            log_var = log_var[~nans_idx]
            precision = precision[~nans_idx]
            target = target[~nans_idx]

        # --- Compute NLL and return ---
        loss = torch.mean(0.5 * precision * (target-mean)**2 + 0.5 * log_var)
        return loss

### -------------------------------------------------------------------------------------------------------- ###
### -------------------- Neg Log-likelihood Bernoulli-Gamma Loss ------------------------------------------- ###
class NLLBerGammaLoss(nn.Module):

    """
    Negative Log-Likelihood Bernoulli-Gamma loss.
    Purpose: Computes NLL for Bernoulli-Gamma distribution, optionally ignoring NaNs.
    Parameters:
        ignore_nans (bool): Ignore NaNs in target domain.
        threshold (float): Threshold for wet days.
    """

    def __init__(self, ignore_nans: bool, threshold: float | None = None) -> None:
        super(NLLBerGammaLoss, self).__init__()
        self.ignore_nans = ignore_nans
        self.threshold = threshold

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Computes NLL Bernoulli-Gamma loss between target and output.
        Parameters:
            target (torch.Tensor): Target data.
            output (torch.Tensor): Model output (p, shape, scale).
        Returns:
            torch.Tensor: Loss value.
        """
        
        # --- Ensure same shape compatibility ---
        assert output.shape[1] == 3, f"Expected P=3 (p, shape, scale), got {output.shape[1]}"
        
        # --- Handle both spatial (H, W) and flattened (GP) shapes ---
        if target.ndim > 3:
            B, C, H, W = target.shape
            target = target.reshape(B, C, -1)
        else:
            B, C, GP = target.shape
        target = target.squeeze() # (B, H*W)
        if output.ndim > 3:
            B, P, H, W = output.shape
            output = output.reshape(B, P, -1)
        else:
            B, P, GP = output.shape
        # --- Split probability, shape and scale Gamma parameters --- #
        p = output[:,0,:].squeeze() # From shape: (B, P=3, H*W) to shape: (B, H*W)
        shape = torch.exp(output[:,1,:]).squeeze() # From shape: (B, P=3, H*W) to shape: (B, H*W)
        scale = torch.exp(output[:,2,:]).squeeze() # From shape: (B, P=3, H*W) to shape: (B, H*W)

        # --- Shift target? --- 
        if self.threshold is not None:
            target = target - self.threshold
            target[target < 0] = 0 

        # ---  Remove Nans if present --- 
        if self.ignore_nans:
            nans_idx = torch.isnan(target)
            p = p[~nans_idx]
            shape = shape[~nans_idx]
            scale = scale[~nans_idx]
            target = target[~nans_idx]

        # --- Compute NLL and return --- 
        bool_rain = torch.greater(target, 0).type(torch.float32)
        epsilon = 0.000001
        noRainCase = (1 - bool_rain) * torch.log(1 - p + epsilon)
        rainCase = bool_rain * (torch.log(p + epsilon) +
                            (shape - 1) * torch.log(target + epsilon) -
                            shape * torch.log(scale + epsilon) -
                            torch.lgamma(shape + epsilon) -
                            target / (scale + epsilon))
        loss = -torch.mean(noRainCase + rainCase)
        return loss



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
                z = zarr.open(ref_path, mode="r")
                idx = z.attrs["variables"][var]
                data = np.array(z[:,idx,:]).squeeze() # From (B, C, GP) to (B, GP)
                dates = [str(date)[:4] for date in z.attrs["dates"]]  
            elif ref_path[-3:] == ".nc":
                z = xr.open_dataset(ref_path)[var]
                data = z.values
                dates = [str(date)[:4] for date in z.time.values]  
                data = data.reshape(len(dates), -1)
            print("⏳ ASYM Loss: Estimating Gamma parameters")
            self.compute_parameters(data, dates, type=type)
            print(f"📦 ASYM Loss: Gamma parameters saved at: {asym_path}")

        # --- Prepare shape, scale and loc ---
        shape, scale, loc = self.load_parameters()
        print(f"📦 ASYM Loss: Gamma parameters loaded from {asym_path}")
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

### -------------------------------------------------------------------------------- ###
### -------------------- Weighted Denoising Score Matching Loss -------------------- ###
class WeightedDenoisingScoreMatchingLoss(nn.Module):
    """
    Weighted Denoising Score Matching loss for diffusion models.
    Purpose: Computes DSM loss for denoising tasks.
    Parameters:
        ignore_nans (bool): Ignore NaNs in target domain.
        sigma_data (float): Data noise level.
    """

    def __init__(self, ignore_nans: bool = False, sigma_data: float = 0.5) -> None:   
        super().__init__()
        self.ignore_nans = ignore_nans
        self.sigma_data = sigma_data

    def c_skip(self, sigma_t):
        """
        Computes skip coefficient for DSM loss.
        Parameters:
            sigma_t: Noise level tensor.
        Returns:
            float: Skip coefficient.
        """
        return self.sigma_data ** 2 / (self.sigma_data ** 2 + sigma_t ** 2)

    def c_out(self, sigma_t):
        """
        Computes output coefficient for DSM loss.
        Parameters:
            sigma_t: Noise level tensor.
        Returns:
            float: Output coefficient.
        """
        return self.sigma_data * sigma_t / (self.sigma_data ** 2 + sigma_t ** 2) ** (0.5)

    def forward(self, target: torch.Tensor, output: torch.Tensor, sigma_t: torch.Tensor, r_t: torch.Tensor = None) -> torch.Tensor:
        """
        Computes DSM loss between target and output.
        Parameters:
            target (torch.Tensor): Target data.
            output (torch.Tensor): Model output.
            sigma_t (torch.Tensor): Noise level.
            r_t (torch.Tensor): Reference tensor.
        Returns:
            torch.Tensor: Loss value.
        """
        
        # Rescale target since we are computing the loss against the raw denoiser output, i.e., against F and not D in the Karras et al., 
        target_rescaled = (target - self.c_skip(sigma_t) * r_t) / self.c_out(sigma_t)

        # Weight
        weight = ((self.sigma_data ** 2 + sigma_t ** 2) / (self.sigma_data ** 2 * sigma_t ** 2))

        # Compute loss of the denoiser
        loss_dsm = weight * (output - target_rescaled) ** 2
        if self.ignore_nans:
            loss_dsm = loss_dsm[~torch.isnan(loss_dsm)]
        loss_dsm = loss_dsm.mean()

        # Return loss
        return loss_dsm 

### -------------------------------------------------------------------------------- ###
### -------------------- Quantised MSE Loss ---------------------------------------- ###
class QuantisedMSELoss(nn.Module):
    """
    Quantised MSE (QMSE) loss plus standard MSE.
    Purpose: Computes QMSE and MSE for quantile bins.
    Parameters:
        zarr_path (str): Path to Zarr store.
        var (str): Target variable.
        alpha (float): Weight for QMSE term.
        n_quantiles (int): Number of quantile bins.
        threshold (float): Threshold for NaN conversion.
        ignore_nans (bool): Ignore NaNs in computations.
    """

    def __init__(self, 
                 zarr_path: str,
                 var: str,
                 alpha: float = 1.0,
                 n_quantiles: int = 10,
                 threshold: float = None,
                 ignore_nans: bool = True):

        super().__init__()

        self.alpha = alpha
        self.n_quantiles = n_quantiles
        self.threshold = threshold
        self.ignore_nans = ignore_nans

        # Store inner MSE object
        self.mse = MseLoss(ignore_nans=ignore_nans)

        # -------- Load reference data from Zarr ----------
        z = zarr.open(zarr_path, mode="r")
        var_idx = z.attrs["variables"][var]
        data = np.array(z[:,var_idx,:]).flatten()  # From (B, C, GP) to (B*GP)
        data = data[~np.isnan(data)]  # remove NaNs if present

        # -------- Compute bin edges from quantiles --------
        self.bin_edges = torch.tensor(
            np.quantile(data, np.linspace(0, 1, n_quantiles + 1)),
            dtype=torch.float32
        )

    # ------------------------------------------------------------
    # Compute QMSE for a batch
    # ------------------------------------------------------------
    def _compute_qmse(self, target: torch.Tensor, output: torch.Tensor):
        """
        Computes QMSE for a batch.
        Parameters:
            target (torch.Tensor): Target data.
            output (torch.Tensor): Model output.
        Returns:
            torch.Tensor: QMSE value.
        """

        # ----- reshape (B,C,H,W) → (B*C*G) -----
        if target.ndim > 3:
            B, C, H, W = target.shape
            target = target.reshape(B, C, -1)
            output = output.reshape(B, C, -1)

        # Flatten completely
        target = target.reshape(-1)
        output = output.reshape(-1)

        # Remove NaNs if required
        if self.ignore_nans:
            mask = ~torch.isnan(target)
            target = target[mask]
            output = output[mask]

        device = target.device
        edges = self.bin_edges.to(device)

        # Assign bins: bucketize returns index in [1 .. n_bins]
        bin_idx = torch.bucketize(target, edges, right=False) - 1
        bin_idx = torch.clamp(bin_idx, 0, self.n_quantiles - 1)

        qmse_terms = []

        # -------- Compute QMSE over bins --------
        for k in range(self.n_quantiles):
            mask_k = (bin_idx == k)
            freq = mask_k.sum()

            if freq == 0:
                continue

            errors_k = (target[mask_k] - output[mask_k]) ** 2
            mse_k = torch.mean(errors_k)

            weight_k = 1.0 / freq.float()
            qmse_terms.append(weight_k * mse_k)

        if len(qmse_terms) == 0:
            return torch.tensor(0.0, device=device)

        return torch.mean(torch.stack(qmse_terms))

    # ------------------------------------------------------------
    # Final Loss = MSE + α * QMSE
    # ------------------------------------------------------------
    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Computes combined MSE and QMSE loss.
        Parameters:
            target (torch.Tensor): Target data.
            output (torch.Tensor): Model output.
        Returns:
            torch.Tensor: Loss value.
        """
        # ------------------------------------------------------------
        # Assert that C = 1 (univariate regression)
        # ------------------------------------------------------------
        if target.ndim == 3:
            B, C, G = target.shape
            assert C == 1, f"Only univariate regression is supported (C=1), got C={C}"
        elif target.ndim == 4:
            B, C, H, W = target.shape
            assert C == 1, f"Only univariate regression is supported (C=1), got C={C}"
        else:
            raise ValueError(
                f"Target tensor must have shape (B, 1, G) or (B, 1, H, W). Got shape: {target.shape}"
            )

        # Same check for output for safety
        if output.ndim == 3:
            B, C, G = output.shape
            assert C == 1, f"Output tensor must have C=1, got C={C}"
        elif output.ndim == 4:
            B, C, H, W = output.shape
            assert C == 1, f"Output tensor must have C=1, got C={C}"
        else:
            raise ValueError(
                f"Output tensor must have shape (B, 1, G) or (B, 1, H, W). Got shape: {output.shape}"
            )

        # ------------------------------------------------------------
        # Convert values below threshold to NaN.
        # ------------------------------------------------------------
        if self.threshold is not None:
                target[target<self.threshold] = np.nan

        # ------------------------------------------------------------
        # Compute loss
        # ------------------------------------------------------------
        mse_val = self.mse(target, output)
        qmse_val = self._compute_qmse(target, output)
        return mse_val + self.alpha * qmse_val



class BinaryCrossEntropyLoss(nn.Module):
    """
    N-dimensional Binary Cross Entropy loss.
    Purpose: Computes BCE for binary targets, supports NaN masking.
    Parameters:
        threshold (float): Threshold for binarization.
        ignore_nans (bool): Ignore NaNs in target domain.
    """

    def __init__(self, threshold: 1.0, ignore_nans: bool = False):
        super().__init__()
        # Allow single float or list/tuple of floats
        if isinstance(threshold, (float, int)):
            self.threshold = threshold
        elif isinstance(threshold, (list, tuple)):
            self.threshold = torch.tensor(threshold, dtype=torch.float32)
        else:
            raise TypeError(
                f"threshold must be float, list, or tuple, got {type(threshold)}"
            )
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Computes BCE loss between target and output.
        Parameters:
            target (torch.Tensor): Target data.
            output (torch.Tensor): Model output.
        Returns:
            torch.Tensor: Loss value.
        """

        # --- Check shapes match -----------------------------
        if target.shape != output.shape:
            raise ValueError(
                f"Target and output must have the same shape, got "
                f"target={target.shape}, output={output.shape}"
            )

        # --- Reshape spatial or graph dimensions ------------
        # Target/output become (B, C, G)
        if target.ndim == 4:
            # (B, C, H, W) → (B, C, H*W)
            B, C, H, W = target.shape
            target = target.reshape(B, C, -1)
            output = output.reshape(B, C, -1)
        elif target.ndim == 3:
            B, C, G = target.shape
        else:
            raise ValueError(
                f"Unsupported shape {target.shape}. Must be (B,C,G) or (B,C,H,W)."
            )

        # --- Broadcast threshold(s) to (1, C, 1) ------------------------------
        if isinstance(self.threshold, torch.Tensor):
            # Per-channel thresholds
            if self.threshold.numel() != C:
                raise ValueError(
                    f"Threshold list has {self.threshold.numel()} entries, "
                    f"but input has C={C} channels."
                )
            thr = self.threshold.view(1, C, 1).to(target.device)
        else:
            # Single float threshold for all channels
            thr = torch.tensor(self.threshold, dtype=torch.float32,
                               device=target.device).view(1, 1, 1)

        # --- Binarize target using thresholds ---------------------------------
        target_bin = (target >= thr).float()

        # --- Flatten to vectors per channel ------------------
        # Shape: (B*C*G,)
        target_flat = target.reshape(-1).float()
        output_flat = output.reshape(-1)

        # --- Handle NaNs (ignore them) -----------------------------
        if self.ignore_nans:
            mask = ~torch.isnan(target_flat)
            output_flat = output_flat[mask]
            target_flat = target_flat[mask]

        # --- Cross-entropy expects class indices (long) -----------------------------
        target_flat = target_flat.long()

        # --- Bernoulli NLL = Binary cross-entropy w/ logits ---
        loss = F.binary_cross_entropy_with_logits(output_flat, target_flat)

        return loss


class BernoulliFocalLoss(nn.Module):
    """
    N-dimensional Bernoulli Focal Loss.
    Purpose: Computes focal loss for binary targets, supports NaN masking.
    Parameters:
        gamma (float): Focusing parameter.
        alpha (float): Weight for positive class.
        threshold (float): Threshold for binarization.
        ignore_nans (bool): Ignore NaNs in target domain.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, threshold: float = 1.0, ignore_nans: bool = False):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.threshold = threshold
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Computes Bernoulli focal loss between target and output.
        Parameters:
            target (torch.Tensor): Target data.
            output (torch.Tensor): Model output.
        Returns:
            torch.Tensor: Loss value.
        """

        if target.shape != output.shape:
            raise ValueError(f"Target and output must match shapes, got {target.shape} vs {output.shape}")

        # --- Reshape to (B, C, G) ----------------------------------------
        if target.ndim == 4:
            B, C, H, W = target.shape
            target = target.reshape(B, C, -1)
            output = output.reshape(B, C, -1)
        elif target.ndim == 3:
            B, C, G = target.shape
        else:
            raise ValueError(f"Unsupported shape {target.shape}. Must be (B,C,G) or (B,C,H,W)")

        # --- Binarize target using the same threshold for all channels ---
        target_bin = (target >= self.threshold).float()

        # --- Flatten for computation --------------------------------------
        target_flat = target_bin.reshape(-1)
        output_flat = output.reshape(-1)

        # --- NaN masking --------------------------------------------------
        if self.ignore_nans:
            mask = ~torch.isnan(target_flat)
            target_flat = target_flat[mask]
            output_flat = output_flat[mask]

        # --- Compute probabilities ---------------------------------------
        p = torch.sigmoid(output_flat)
        pt = p * target_flat + (1 - p) * (1 - target_flat)

        # --- Compute focal weight and alpha_t ----------------------------------------
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = self.alpha * target_flat + (1 - self.alpha) * (1 - target_flat)

        # --- Compute focal loss ------------------------------------------
        loss = -alpha_t * focal_weight * torch.log(pt + 1e-8)

        return loss.mean()


class CRPSSpectralLoss(nn.Module):

    """
    Fair Continuous Ranked Probability Score (CRPS) with spectral component.
    Purpose: Computes CRPS and spectral CRPS for ensemble predictions.
    Parameters:
        ignore_nans (bool): Ignore NaNs in target domain.
        H_shape (int): Height of spatial domain.
        W_shape (int): Width of spatial domain.
        beta (int): Power parameter for CRPS.
        lambda_spectral (float): Weight for spectral CRPS.
        spatial_resolution (float): Spatial resolution for filtering.
    """

    def __init__(self, ignore_nans: bool,
                 H_shape: int, W_shape: int, 
                 beta: int = 1,
                 lambda_spectral: float = 0.1,
                 spatial_resolution: float = None) -> None:
        super(CRPSSpectralLoss, self).__init__()
        self.ignore_nans = ignore_nans
        self.H_shape = H_shape
        self.W_shape = W_shape
        self.beta = beta
        self.lambda_spectral = lambda_spectral
        if spatial_resolution is not None and spatial_resolution <= 0:
            raise ValueError("spatial_resolution must be > 0 when provided.")
        self.spatial_resolution = spatial_resolution
        self.filter_nans = False # Control whether to filter out nans in _CRPS_pointwise

    def _CRPS_pointwise(self, target: torch.Tensor, output) -> torch.Tensor:
        """
        Computes pointwise CRPS for ensemble predictions.
        Parameters:
            target (torch.Tensor): Target data.
            output (list): List of ensemble predictions.
        Returns:
            torch.Tensor: CRPS value.
        """
        
        if self.ignore_nans and self.filter_nans:
            nans_idx = torch.isnan(target)
            target = target[~nans_idx]
            output = [out[~nans_idx] for out in output]

        # Number of ensemble members
        M = len(output)

        # Error between target and each prediction
        first_term = 0.0
        for i in range(M):
            first_term += torch.abs(target - output[i]) ** self.beta
        first_term = first_term / M

        # Difference between all pairs of predictions
        if M > 1:
            second_term = 0.0
            for i in range(M):
                for j in range(M):
                    second_term += torch.abs(output[i] - output[j]) ** self.beta
            second_term = second_term / (2*M*(M-1)) # Fair CRPS
        else:
            second_term = 0.0

        # Final loss
        loss = torch.mean(first_term - second_term)

        return loss

    def _FFT(self, data: torch.Tensor) -> torch.Tensor:
        """
        Computes FFT for input data, applies low-pass filtering if needed.
        Parameters:
            data (torch.Tensor): Input data.
        Returns:
            torch.Tensor: FFT-transformed data.
        """

        # It does not make sense to filter out nans in the spectral domain
        self.filter_nans = False

        # Fill nans with 0 for the FFT computation
        if isinstance(data, torch.Tensor): # For the target
            data = [torch.nan_to_num(data, nan=0.0)]
        else:
            data = [torch.nan_to_num(d, nan=0.0) for d in data]

        B = data[0].shape[0] # Batch size
        if data[0].ndim == 3: M = data[0].shape[1] # Number of ensemble members

        # Reshape to spatial dimensions
        if data[0].ndim == 2:
            data = [member.view(B, self.H_shape, self.W_shape) for member in data]
        elif data[0].ndim == 3:
            data = [member.view(B, M, self.H_shape, self.W_shape) for member in data]

        # Compute FFT
        data = [torch.fft.rfft2(member) for member in data]

        # Optionally remove frequencies beyond the Nyquist limit
        if self.spatial_resolution is not None:
            k_nyquist = 2.0 * torch.pi / (2.0 * self.spatial_resolution)
            kx = 2.0 * torch.pi * torch.fft.rfftfreq(self.W_shape, d=self.spatial_resolution)
            ky = 2.0 * torch.pi * torch.fft.fftfreq(self.H_shape, d=self.spatial_resolution)
            k_radius = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
            low_pass_mask = k_radius <= k_nyquist
            low_pass_mask = low_pass_mask.to(device=data[0].device)
            data = [member * low_pass_mask for member in data]

        return data
        
    def forward(self, target: torch.Tensor, output) -> torch.Tensor:
        """
        Computes combined CRPS and spectral CRPS loss.
        Parameters:
            target (torch.Tensor): Target data.
            output (list or torch.Tensor): Ensemble predictions.
        Returns:
            torch.Tensor: Loss value.
        """

        if isinstance(output, torch.Tensor):
            output = [output]

        # Compute standard CRPS
        self.filter_nans = True
        crps_field = self._CRPS_pointwise(target, output)

        # Compute spectral CRPS
        target_fft = self._FFT(target)[0]
        output_fft = self._FFT(output)
        crps_spectral = self._CRPS_pointwise(target_fft, output_fft)

        # Compute total loss
        loss = crps_field + self.lambda_spectral * crps_spectral
        return loss