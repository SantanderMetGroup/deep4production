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
    Standard Mean Square Error (MSE). It is possible to compute
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
        super(MseLoss, self).__init__()
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        
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
    Negative Log-Likelihood of a Gaussian distribution. It is possible to compute
    this metric over a target dataset with nans.

    Notes
    -----
    Expects model output with separate mean and log-variance per channel.

    Parameters
    ----------
    ignore_nans : bool
        Whether to allow the loss function to ignore nans in the
        target domain.

    target : torch.Tensor
        Target/ground-truth data.
        Shape: (B, C, H, W) or (B, C, GP)

    output : torch.Tensor
        Predicted data (model's output). 
        Shape: (B, C, P, H, W) or (B, C, P, GP)
            where P = 2 (mean, log_var)
            GP = gridpoint
    """

    def __init__(self, ignore_nans: bool) -> None:
        super(NLLGaussianLoss, self).__init__()
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:

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
    Negative Log-Likelihood of a Bernoulli-gamma distributions. It is possible to compute
    this metric over a target dataset with nans.

    Notes
    -----
    This loss function needs as input three values, corresponding to the p, shape
    and scale parameters. THese must be provided concatenated as an unique vector.

    Parameters
    ----------
    ignore_nans : bool
        Whether to allow the loss function to ignore nans in the
        target domain.

    target : torch.Tensor
        Target/ground-truth data

    output : torch.Tensor
        Predicted data (model's output). This vector must be composed
        by the concatenation of the predicted p, shape and scale.
    """

    def __init__(self, ignore_nans: bool, threshold: float | None = None) -> None:
        super(NLLBerGammaLoss, self).__init__()
        self.ignore_nans = ignore_nans
        self.threshold = threshold

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        
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
    Generalization of the asymmetric loss function tailored for daily precipitation developed in
    Doury et al. 2024. It is possible to compute this metric over a target dataset
    with nans.

    Doury, A., Somot, S. & Gadat, S. On the suitability of a convolutional neural
    network based RCM-emulator for fine spatio-temporal precipitation. Clim Dyn (2024).
    https://doi.org/10.1007/s00382-024-07350-8

    Notes
    -----
    This loss function relies on gamma distribution fitted for each gridpoint in the
    spatial domain. This class provides all the methods require to fit these 
    distributions to the data.

    The level of asymmetry can be adjusted by the pairs asym_weight and cdf_pow.

    Parameters
    ----------
    ignore_nans : bool
        Whether to allow the loss function to ignore nans in the
        target domain.

    asym_weight : float, optional
        Weight for the asymmetric term at the loss function relative to the MAE term.
        Default value: 1 (as in Doury et al., 2024)

    cdf_pow : float, optional
        Pow for the CDF at the asymmetric term of the loss function.
        Default value: 2 (as in Doury et al., 2024)
        Higher values make a bigger differentiation between the weight for high/low percentiles

    asym_path : str
        Path to the folder to save the fitted distributions.

    appendix : str, optional
        String to add to the files generated/loaded for this loss function.
        (e.g., appendix=test1 -> scale_test1.npy). If not provided no appendix
        will be added.

    target : torch.Tensor
        Target/ground-truth data

    output : torch.Tensor
        Predicted data (model's output). This vector must be composed
        by the concatenation of the predicted mean and logarithm of the
        variance.
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
        Check for the existence of the gamma distributions
        """

        shape_exist = os.path.exists(self.shape_path)
        scale_exist = os.path.exists(self.scale_path)
        loc_exist = os.path.exists(self.loc_path)
        return (shape_exist and scale_exist and loc_exist)

    def load_parameters(self):
        """
        Load the gamma distributions from asym_path.
        """

        shape = np.load(self.shape_path)
        scale = np.load(self.scale_path)
        loc = np.load(self.loc_path)
        return shape, scale, loc

    def _compute_gamma_parameters(self, x: np.ndarray, threshold: float=1.0) -> tuple:

        """
        Fit a gamma distribution to the wet days of the provided
        1D np.ndarray.

        Parameters
        ----------      
        x : np.ndarray
            1D np.ndarray containing the precipitation values across time
            for a specific gridpoint.com

        Returns
        -------
        tuple
        The shape, loc and scale parameters of the fitted gamma
        distribution.
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
        Iterate over the xr.Dataset and compute for each spatial gridpoint
        the parameters of a fitted gamma distribution for the wet days.

        Parameters
        ----------      
        data : ... (2D np.array) --> dimensions BxGP (batch, sgridpoint)

        var_target : str
            Target variable.
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

        # # --- Fit a Gamma distribution (per year) ---
        # gamma_params = []
        # group_years = data.groupby('time.year')
        # for year, group in group_years:
        #     print(f'Year: {year}')
        #     y_year = group[var_target].values
        #     params_year = np.apply_along_axis(self._compute_gamma_parameters, axis=0, arr=y_year) # shape, loc, scale
        #     gamma_params.append(params_year)
        # gamma_params = np.nanmean(np.stack(gamma_params), axis=0)
        # shape = gamma_params[0, :]
        # scale = gamma_params[2, :]
        # loc = gamma_params[1, :]

        # --- Save the parameters in the asym_path ---
        np.save(file=self.shape_path, arr=shape)
        np.save(file=self.scale_path, arr=scale)
        np.save(file=self.loc_path, arr=loc)

    def prepare_parameters(self, shape, scale, loc):
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
        Compute the value of the cumulative distribution function (CDF) for
        the data.

        Parameters
        ----------      
        data : torch.Tensor
            Data (from the target dataset) to compute the CDF for.
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
        Compute the loss function for the target and output data
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
    def __init__(self, ignore_nans: bool = False, sigma_data: float = 0.5) -> None:   
        super().__init__()
        self.ignore_nans = ignore_nans
        self.sigma_data = sigma_data

    def c_skip(self, sigma_t):
        return self.sigma_data ** 2 / (self.sigma_data ** 2 + sigma_t ** 2)

    def c_out(self, sigma_t):
        return self.sigma_data * sigma_t / (self.sigma_data ** 2 + sigma_t ** 2) ** (0.5)

    def forward(self, target: torch.Tensor, output: torch.Tensor, sigma_t: torch.Tensor, r_t: torch.Tensor = None) -> torch.Tensor:
        
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
    Implements the Quantised MSE (QMSE) + standard MSE (with coefficient α).

    Parameters
    ----------
    zarr_path : str
        Path to Zarr store containing training target data.
    alpha : float
        Weight multiplying the QMSE term in the final combined loss.
    n_quantiles : int
        Number of quantile bins for QMSE.
    ignore_nans : bool
        Whether to ignore NaNs in both MSE and QMSE computations.
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
    N-dimensional Cross Entropy Loss supporting:
    - Shapes (B, C, G)  or (B, C, H, W)
    - Multivariate cases (C >= 1)
    - Optional NaN masking in the target

    Parameters
    ----------
    ignore_nans : bool
        If True, locations where target = nan will be ignored.
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
        Parameters
        ----------
        output : torch.Tensor
            Logits of shape (B, C, G) or (B, C, H, W)
        target : torch.Tensor
            Binary labels in {0,1}, same shape as output
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
    N-dimensional Bernoulli Focal Loss (binary Focal Loss).

    Supports:
      - Shapes (B, C, G) or (B, C, H, W)
      - Multivariate channels (C >= 1)
      - Optional NaN masking
      - Same gamma, alpha, and threshold for all channels

    Parameters
    ----------
    gamma : float
        Focusing parameter γ ≥ 0.
    alpha : float
        Weight for positive class (0 < alpha < 1)
    threshold : float
        Threshold to binarize targets
    ignore_nans : bool
        Whether to ignore NaNs in target.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, threshold: float = 1.0, ignore_nans: bool = False):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.threshold = threshold
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        output : torch.Tensor
            Raw output data of shape (B, C, G) or (B, C, H, W)
        target : torch.Tensor
            Raw target data, same shape as output, will be binarized.
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
class PriceLoss(nn.Module):

    """
    L = |y - y'| ⊙ (y + 1)
    where y' is the predicted output, y is the target, and ⊙ represents
    elementwise multiplication across space.

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
        super(PriceLoss, self).__init__()
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:

        if target.ndim > 2:
            target = target.reshape(target.shape[0], -1)
            output = output.reshape(output.shape[0], -1)

        if self.ignore_nans:
            nans_idx = torch.isnan(target)
            output = output[~nans_idx]
            target = target[~nans_idx]

        loss = torch.mean(torch.abs(target - output) * (target + 1))
        return loss

class DualOutputLoss(nn.Module):

    """
    Combined loss function for DeepESDDualOutput model.
    
    This loss function combines:
    - Binary Cross Entropy for dry/wet day classification
    - Loss for precipitation amount prediction
    
    The target data is split based on a threshold (similar to BerGamma loss):
    - Classification target: 1 if precipitation > threshold, 0 otherwise
    - Amount target: original precipitation values (only used where classification = 1)
    
    Parameters
    ----------
    ignore_nans : bool
        Whether to allow the loss function to ignore nans in the
        target domain.
        
    threshold : float
        Threshold for dry/wet day classification. Values above this threshold
        are considered wet days.
        
    classification_weight : float, optional
        Weight for the classification loss component. Default is 1.0.
        
    amount_weight : float, optional
        Weight for the amount loss component. Default is 1.0.
    """

    def __init__(self, ignore_nans: bool, threshold: float = 0.0, 
                 classification_weight: float = 1.0, amount_weight: float = 1.0) -> None:
        super(DualOutputLoss, self).__init__()
        self.ignore_nans = ignore_nans
        self.threshold = threshold
        self.classification_weight = classification_weight
        self.amount_weight = amount_weight
        
        # Initialize BCE loss
        self.bce_loss = nn.BCELoss(reduction='none')
        
        # Pre-allocate tensors to avoid memory allocation during forward pass
        self._zero_tensor = None

    def forward(self, target: torch.Tensor, output: dict) -> torch.Tensor:

        # Extract outputs from the dual-output model
        classification_pred = output[:, :output.shape[1]//2]
        amount_pred = output[:, output.shape[1]//2:]
            
        # Create classification target (1 if > threshold, 0 otherwise)
        classification_target = (target > self.threshold).float()
        
        # Handle NaNs if needed
        if self.ignore_nans:
            nans_idx = torch.isnan(target)
            classification_pred = classification_pred[~nans_idx]
            classification_target = classification_target[~nans_idx]
            amount_pred = amount_pred[~nans_idx]
            target = target[~nans_idx]
        
        # Classification loss (Binary Cross Entropy)
        bce_losses = self.bce_loss(classification_pred, classification_target)
        classification_loss = torch.mean(bce_losses)

        # Amount loss (only for wet days)
        wet_mask = (classification_target == 1)
        
        # Check if there are any wet days without creating intermediate tensors
        num_wet_days = wet_mask.sum()
        
        if num_wet_days > 0:  # If there are any wet days
            ############################################
            # Price loss formula
            # diff = torch.abs(target - amount_pred)
            # weighted_diff = diff * (target + 1)
            ############################################

            ############################################
            # Non-parametric asymmetric loss formula
            # diff = torch.abs(target - amount_pred)
            # weighted_diff = diff * torch.max(torch.tensor(1.0), target - amount_pred)
            
            # Only compute mean for wet days
            # amount_loss = weighted_diff[wet_mask].mean()
            ############################################

            ############################################
            diff = ((target ** 0.3) - (amount_pred)) ** 2
            amount_loss = diff[wet_mask].mean()
            ############################################
        else:
            # If no wet days, amount loss is 0
            if self._zero_tensor is None or self._zero_tensor.device != target.device:
                self._zero_tensor = torch.tensor(0.0, device=target.device, requires_grad=True)
            amount_loss = self._zero_tensor
        
        # Combine losses
        total_loss = (self.classification_weight * classification_loss + 
                      self.amount_weight * amount_loss)
        
        return total_loss


# ------------------------------------------------------------------------------------------------------------------------
class CRPSSpectralLoss(nn.Module):
    """
    CRPS + Spectral CRPS loss for statistical downscaling.

    Supports:
        target: (B, C, G) or (B, C, H, W)
        output: (B, M, C, G) or (B, M, C, H, W)
    
     Loss:
        L = CRPS_point(output, target) 
            + lambda_freq * CRPS_point( FFT(output), FFT(target) )
    """

    def __init__(self, lambda_freq=0.1, lowpass_ratio=0.25, H=None, W=None):
        super().__init__()
        self.lambda_freq = lambda_freq
        self.eps_ratio = 0.05 
        self.lowpass_ratio = lowpass_ratio
        self.H = H
        self.W = W


    # ---------------------------------------------------------
    def _crps_pointwise(self, pred, target):
        """
        pred:   (B, M, C, G)
        target: (B, C, G)
        """
        B, M, C, G = pred.shape
        eps = self.eps_ratio / M  # epsilon = 0.05 / M

        # Expand target across ensemble dimension
        target_exp = target.unsqueeze(1)  # (B, 1, C, G)

        # ---- Term 1: MAE between ensemble member & truth ----
        mae = torch.abs(pred - target_exp).mean(dim=1)  # (B, C, G)
        term1 = mae.mean()

        # ---- Term 2: ensemble spread ----
        pred_i = pred.unsqueeze(2)  # (B, M, 1, C, G)
        pred_j = pred.unsqueeze(1)  # (B, 1, M, C, G)
        pairwise = torch.abs(pred_i - pred_j)  # (B, M, M, C, G)

        # Remove diagonal safely
        diag = torch.eye(M, device=pred.device).bool().view(1, M, M, 1, 1)
        pairwise = pairwise.masked_fill(diag, 0.0)  # zero out diagonal
        spread = pairwise.sum(dim=2) / (M - 1)      # mean over other members
        term2 = spread.mean() * (1 - eps)

        return term1 - 0.5 * term2

    # ---------------------------------------------------------
    def _lowpass_fft(self, x):
        """
        Apply 2D rFFT over spatial dimensions (H, W) with low-pass filtering.

        x: (B, M, C, G) or (B, M, C, H, W)
        - If G, assumes self.H and self.W are set and G = H*W
        Returns: (B, M, C, H_freq, W_freq) filtered FFT
        """

        # --- Reshape flattened G to (H, W) if needed ---
        if x.ndim == 4:  # (B, M, C, G)
            B, M, C, G = x.shape
            assert G == self.H * self.W, "G must equal H*W"
            x = x.reshape(B, M, C, self.H, self.W)

        # --- Compute 2D rFFT along spatial dims ---
        X = torch.fft.rfft2(x, dim=(-2, -1))  # shape: (B, M, C, H, W_freq)
        H, W_freq = X.shape[-2], X.shape[-1]

        # --- Build low-pass mask ---
        cut_h = max(int(H * self.lowpass_ratio), 1)
        cut_w = max(int(W_freq * self.lowpass_ratio), 1)
        mask = torch.zeros_like(X, dtype=torch.bool)
        mask[..., :cut_h, :cut_w] = True

        # --- Apply mask ---
        X_filtered = X * mask
        return X_filtered

    # ---------------------------------------------------------
    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        target: (B, C, G) or (B, C, H, W)
        output: (B, M, C, G) or (B, M, C, H, W)
        """

        # --- Handle both spatial (H, W) and flattened (GP) shapes ---
        if target.ndim > 3: # stack spatial dimensions
            B, C, H, W = target.shape
            target = target.reshape(B, C, -1) # From shape: (B, C, H, W) to (B, C, H*W)
        if output.ndim > 3: # stack spatial dimensions
            B, M, C, H, W = output.shape
            output = output.reshape(B, M, C, -1) # From shape: (B, C, H, W) to (B, C, H*W)
        
        # # --- Remove Nans if present ---
        # if self.ignore_nans:
        #     nans_idx = torch.isnan(target)
        #     output = output[~nans_idx]
        #     target = target[~nans_idx]
        
        # ---------------- Pointwise CRPS ----------------
        crps_p = self._crps_pointwise(output, target) # (B, C, G) → scalar

        # ---------------- Spectral CRPS -----------------
        output_fft = self._lowpass_fft(output).reshape(B, M, C, -1)
        # print(output_fft.shape)
        target_fft = self._lowpass_fft(target.unsqueeze(1)).reshape(B, C, -1)  # (B, M, C, G)
        # print(target_fft.shape)
        crps_f = self._crps_pointwise(output_fft.abs(), target_fft.abs())

        # ---------------- Total Loss --------------------
        loss = crps_p + self.lambda_freq * crps_f
        return loss