"""
Diffusion model loss functions (EDM-weighted denoising score matching).

Authors:
    Jose González-Abad
    Alfonso Hernanz
    Jorge Baño-Medina
"""

import torch
import torch.nn as nn


### -------------------------------------------------------------------------------- ###
### -------------------- Weighted Denoising Score Matching Loss -------------------- ###
class WeightedDenoisingScoreMatchingLoss(nn.Module):
    """
    EDM-weighted denoising score-matching loss (Karras et al. 2022).

        L(theta) = E[ lambda(sigma) * || D_theta - target ||^2 ]
        lambda(sigma) = 1 / c_out^2 = (sigma_data^2 + sigma^2) / (sigma_data * sigma)^2

    Expects `output` to be the *denoised* prediction D_theta (i.e. the backbone
    has already been wrapped by an EDM preconditioner such as
    `deep4production.deep.models.diffusion.edm_precond.EDMPrecond`). This loss
    does NOT apply any c_skip / c_out rescaling itself — preconditioning lives
    inside the preconditioner module, not here.

    Parameters
    ----------
    ignore_nans : bool
        Drop NaN entries from the per-element loss before averaging.
    sigma_data : float
        Data sigma; must match the value used by the EDM preconditioner.
    """

    def __init__(self, ignore_nans: bool = False, sigma_data: float = 0.5) -> None:
        super().__init__()
        self.ignore_nans = ignore_nans
        self.sigma_data = sigma_data

    def forward(
        self, target: torch.Tensor, output: torch.Tensor, sigma_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        target : (B, C, H, W)
            Clean signal (residual r or full field, depending on the trainer).
        output : (B, C, H, W)
            Denoised prediction D_theta from the EDM preconditioner.
        sigma_t : (B, 1, 1, 1) or broadcastable
            Per-sample noise level used when corrupting `target`.

        Returns
        -------
        Scalar loss.
        """
        weight = (self.sigma_data**2 + sigma_t**2) / (self.sigma_data * sigma_t) ** 2
        loss = weight * (output - target) ** 2
        if self.ignore_nans:
            loss = loss[~torch.isnan(loss)]
        return loss.mean()
