"""
Negative log-likelihood loss functions for parametric distributions.

Authors:
    Jose González-Abad
    Alfonso Hernanz
    Jorge Baño-Medina
"""

import torch
import torch.nn as nn


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
        assert (
            output.shape[2] == 2
        ), f"Expected P=2 (mean, log_var), got {output.shape[2]}"

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
        mean = output[:, :, 0, :]  # (B, C, GP)
        log_var = output[:, :, 1, :]  # (B, C, GP)
        precision = torch.exp(-log_var)

        # --- Remove Nans if present ---
        if self.ignore_nans:
            nans_idx = torch.isnan(target)
            mean = mean[~nans_idx]
            log_var = log_var[~nans_idx]
            precision = precision[~nans_idx]
            target = target[~nans_idx]

        # --- Compute NLL and return ---
        loss = torch.mean(0.5 * precision * (target - mean) ** 2 + 0.5 * log_var)
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
        assert (
            output.shape[1] == 3
        ), f"Expected P=3 (p, shape, scale), got {output.shape[1]}"

        # --- Handle both spatial (H, W) and flattened (GP) shapes ---
        if target.ndim > 3:
            B, C, H, W = target.shape
            target = target.reshape(B, C, -1)
        else:
            B, C, GP = target.shape
        target = target.squeeze()  # (B, H*W)
        if output.ndim > 3:
            B, P, H, W = output.shape
            output = output.reshape(B, P, -1)
        else:
            B, P, GP = output.shape
        # --- Split probability, shape and scale Gamma parameters --- #
        p = output[:, 0, :].squeeze()  # From shape: (B, P=3, H*W) to shape: (B, H*W)
        shape = torch.exp(
            output[:, 1, :]
        ).squeeze()  # From shape: (B, P=3, H*W) to shape: (B, H*W)
        scale = torch.exp(
            output[:, 2, :]
        ).squeeze()  # From shape: (B, P=3, H*W) to shape: (B, H*W)

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
        rainCase = bool_rain * (
            torch.log(p + epsilon)
            + (shape - 1) * torch.log(target + epsilon)
            - shape * torch.log(scale + epsilon)
            - torch.lgamma(shape + epsilon)
            - target / (scale + epsilon)
        )
        loss = -torch.mean(noRainCase + rainCase)
        return loss
