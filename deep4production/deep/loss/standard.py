"""
Standard regression loss functions (MAE, MSE, Quantised-MSE).

Authors:
    Jose González-Abad
    Alfonso Hernanz
    Jorge Baño-Medina
"""

import torch
import torch.nn as nn
import numpy as np
from deep4production.utils.zarr import open_zarr_store


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
        z = open_zarr_store(zarr_path, fmt="auto")
        var_idx = z.attrs["variables"][var]
        data = np.array(z["data"][:, var_idx, :]).flatten()  # From (B, C, GP) to (B*GP)
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
