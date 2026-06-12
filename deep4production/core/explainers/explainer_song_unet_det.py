"""
Gradient explainer for the deterministic SongUNet downscaler.

Mirrors the forward of ``downscaler_song_unet_det``: the model conditions
entirely through ``cond_low`` (the normalised predictors), with the noisy-input
slot fed zeros and the noise label fixed at t = 0. Everything else (model /
metadata / normalizer / preprocessing / region machinery) is inherited from
``Explainer``.

Authors
-------
    Jorge Baño-Medina
"""

import torch

from deep4production.core.explainers.explainer import Explainer
from deep4production.utils.log import get_logger

log = get_logger("explainer.songunet")


class ExplainerSongUNetDet(Explainer):
    """Gradient explainer for deterministic SongUNet regressors."""

    def _attribution_forward(self, inp_norm, batch_dates):
        """
        Deterministic SongUNet forward attached to the autograd graph.

        Parameters
        ----------
        inp_norm : torch.Tensor
            (B, C_x, H_x, W_x) normalised predictor leaf tensor.
        batch_dates : list
            Unused (deterministic model conditions only on ``cond_low``); kept
            for signature compatibility with the base ``Explainer``.

        Returns
        -------
        torch.Tensor : (B, C_y, H_y, W_y) prediction in normalised space.
        """
        B = inp_norm.shape[0]
        C_y = len(self.vars_y)
        spatial = [self.H_y, self.W_y] if self.transform_to_2D_y else [self.G_y]
        x_in = torch.zeros(B, C_y, *spatial, device=self.device)
        t = torch.zeros(B, device=self.device)
        return self.model(x=x_in, t=t, cond_low=inp_norm)
