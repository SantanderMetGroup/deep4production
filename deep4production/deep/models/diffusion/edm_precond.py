"""
EDMPrecond: Elucidated Diffusion Model preconditioning wrapper (Karras et al. 2022).

Wraps a raw U-Net backbone F_θ with the EDM preconditioning and returns the
denoised prediction D_θ. Used by CorrDiff-style (residual-diffusion) trainers.

    c_in(σ)    = 1 / √(σ_data² + σ²)
    c_skip(σ)  = σ_data² / (σ_data² + σ²)
    c_out(σ)   = σ_data · σ / √(σ_data² + σ²)
    c_noise(σ) = 0.25 · log(σ)

    D_θ(x, σ; c) = c_skip(σ) · x + c_out(σ) · F_θ(c_in(σ) · x, c_noise(σ), c)

Training loss (EDM):
    L(θ) = E[ λ(σ) · ‖D_θ − target‖² ],  λ(σ) = 1 / c_out² = (σ_d² + σ²) / (σ_d · σ)²

The loss weighting λ(σ) is applied by
`deep4production.deep.loss.WeightedDenoisingScoreMatchingLoss`.

The same preconditioner is used at training AND inference — the sampler calls
`precond(x_t, σ_t, cond_low, cond_high)` exactly as the trainer does. This is
the main reason preconditioning lives in its own nn.Module (and not in the
trainer or the loss): no duplication, and `sigma_data` is stored as a buffer
so the checkpoint is self-contained.

CPMGEM's sub-VP SDE does NOT use this wrapper — that trainer calls the
backbone directly.

Author:
    Jorge Baño-Medina
"""

import torch
import torch.nn as nn

from deep4production.utils.general import get_func_from_string


def build_edm_model(backbone: dict, sigma_data: float = 0.5) -> "EDMPrecond":
    """
    Factory that instantiates a backbone from a nested YAML spec and wraps it
    in an EDMPrecond.

    Enables a single `{module, name, kwargs}` model entry in the YAML (what
    `deep4production.utils.general.get_func_from_string` expects) while still
    allowing the backbone to be fully configurable:

        model_params:
          name: build_edm_model
          module: deep4production.deep.models.diffusion.edm_precond
          kwargs:
            sigma_data: 0.5
            backbone:
              module: deep4production.deep.models.unet.song_unet
              name: SongUNet
              kwargs: { ... }

    The checkpoint metadata stores these same kwargs verbatim, so
    `deep4production.deep.utils.load_model` reconstructs the preconditioner
    identically at inference — no duplicated wiring.

    Parameters
    ----------
    backbone : dict
        {"module": ..., "name": ..., "kwargs": ...} spec for the U-Net backbone.
    sigma_data : float

    Returns
    -------
    EDMPrecond wrapping the instantiated backbone.
    """
    net = get_func_from_string(
        module_string=backbone["module"],
        func_string=backbone["name"],
        kwargs=backbone.get("kwargs", {}),
    )
    return EDMPrecond(backbone=net, sigma_data=sigma_data)


class EDMPrecond(nn.Module):
    """EDM preconditioner wrapping a backbone F_θ.

    Parameters
    ----------
    backbone : nn.Module
        The raw U-Net. Must accept forward(x, t, cond_low=None, cond_high=None)
        where `t` is the scalar noise label (here: c_noise).
    sigma_data : float
        Standard deviation of the data at σ=0. Stored as a buffer so it
        travels with the checkpoint.
    """

    def __init__(self, backbone: nn.Module, sigma_data: float = 0.5) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("sigma_data", torch.tensor(float(sigma_data)))

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        cond_low: torch.Tensor = None,
        cond_high: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)
            Noisy input at noise level σ.
        sigma : (B,) or (B, 1, 1, 1)
            Per-sample noise level.
        cond_low, cond_high : optional conditioning tensors, as accepted by
            the backbone.

        Returns
        -------
        (B, C, H, W) — denoised prediction D_θ.
        """
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1, 1)

        sd2 = self.sigma_data ** 2
        s2 = sigma ** 2
        c_in    = 1.0 / (sd2 + s2).sqrt()
        c_skip  = sd2 / (sd2 + s2)
        c_out   = self.sigma_data * sigma / (sd2 + s2).sqrt()
        c_noise = 0.25 * sigma.log().view(-1)

        F_theta = self.backbone(
            x=c_in * x,
            t=c_noise,
            cond_low=cond_low,
            cond_high=cond_high,
        )
        return c_skip * x + c_out * F_theta
