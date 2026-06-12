"""
Trainer for CPMGEM-residual — the sub-VP × residual cell of the 2×2 study.

This is the cross of the two existing diffusion stacks:

  * dataset / regressor plumbing from ResDiff (trainer_resdiff + pydataset_resdiff):
    the frozen deterministic regressor is run once at init, residuals r = y − ŷ_det
    are cached to a zarr, and the loader serves (r, c_low, c_high) where c_high is
    the regressor mean ŷ_det.
  * the continuous-time sub-VP SDE forward process from CPMGEM (trainer_cpmgem):
    the network predicts the noise ε with an MSE objective, using a *plain* SongUNet
    (no EDM preconditioner).

So we model the **residual** with the **sub-VP formulation** — isolating the
diffusion-formulation axis from the parameterization-target axis.

Residual standardization (important)
------------------------------------
The sub-VP marginal (mean(t) = e^{−½B(t)}, std(t) = 1 − e^{−B(t)}, prior N(0, I))
is calibrated for ~unit-variance clean data. CPMGEM-direct gets that for free
because the target y is min-max rescaled to [−1, 1]. The *residual* of normalized
fields has std ≪ 1, and — unlike EDM, which absorbs arbitrary data scale through
its sigma_data preconditioning — sub-VP has no scale-handling mechanism. Feeding a
low-variance residual into sub-VP mis-calibrates the SNR schedule (over-noised at
every t). We therefore standardize the residual per variable to ~unit variance
using the per-channel mean/std already stored in the residuals zarr (computed by
pydataset_resdiff), run the sub-VP process in standardized space, and record the
stats in metadata so the downscaler can invert (r̂ = r_std·std + mean) before
adding back the regressor mean.

Authors:
    Jorge Baño-Medina
"""

import numpy as np
import torch
import zarr

from deep4production.core.trainers.trainer_resdiff import (
    trainer_custom as trainer_resdiff,
)
from deep4production.utils.log import get_logger

log = get_logger("trainer.cpmgem_residual")


class trainer_custom(trainer_resdiff):
    """
    Trainer for CPMGEM-residual: sub-VP SDE diffusion on the regression residual.

    Inherits dataset / regressor wiring (get_pydatasets, update_metadata) from
    trainer_resdiff and the base training loop from trainer. Overrides
    model_backprop to run the CPMGEM sub-VP forward process on the (optionally
    standardized) residual, and get_pydatasets to additionally cache the residual
    standardization statistics.

    Additional YAML keys under training_params.kwargs:
        noise_params:
            beta_min : float  – β at t=0 (paper: 0.1)
            beta_max : float  – β at t=1 (paper: 20.0)
            t_min    : float  – lower bound for continuous t (default 1e-5)
        standardize_residual : bool – standardize the residual to ~unit variance
            before the sub-VP process (default True; strongly recommended).
    """

    def __init__(
        self,
        data,
        dataloader,
        id_dir,
        model_info,
        graph,
        d4dpy,
        Mlflow,
        normalizer_info_x=None,
        normalizer_info_y=None,
        normalizer_info_f=None,
        hardware=None,
    ):
        super().__init__(
            data=data,
            dataloader=dataloader,
            id_dir=id_dir,
            model_info=model_info,
            graph=graph,
            d4dpy=d4dpy,
            Mlflow=Mlflow,
            normalizer_info_x=normalizer_info_x,
            normalizer_info_y=normalizer_info_y,
            normalizer_info_f=normalizer_info_f,
            hardware=hardware,
        )

        self.standardize_residual = bool(
            model_info["training_params"]["kwargs"].get("standardize_residual", True)
        )
        # (1, C_y, 1, 1) buffers, populated in get_pydatasets from the residuals zarr.
        self._res_mean = None
        self._res_std = None
        log.info(
            "CPMGEM-residual trainer ready (sub-VP SDE on residual; "
            "standardize_residual=%s)",
            self.standardize_residual,
        )

    # ─────────────────────────────────────────────────────────────────────────
    def get_pydatasets(self):
        """
        Build the residual pydatasets (inherited) and additionally read the
        per-variable residual standardization stats from the training residuals
        zarr that pydataset_resdiff just wrote.

        The residuals zarr stores per-channel mean/std over 2·C_y channels ordered
        [*_residual, *_normalized]; we keep the first C_y (the residual channels).
        Stats are stashed as (1, C_y, 1, 1) tensors and recorded in metadata so the
        downscaler can invert the standardization.
        """
        train_dataset, valid_dataset = super().get_pydatasets()

        if self.standardize_residual:
            num_y = len(train_dataset.vars_y)
            # pydataset_resdiff names the training store "<residuals>_training.zarr".
            residuals_path = self.d4dpy["residuals"]["path"]
            zarr_path = f"{residuals_path[:-5]}_training.zarr"
            store = zarr.open(zarr_path, mode="r")
            res_mean = np.asarray(store["mean"][:num_y], dtype=np.float32)
            res_std = np.asarray(store["std"][:num_y], dtype=np.float32)
            # Guard against zero-variance channels (degenerate residual).
            res_std = np.where(res_std > 0, res_std, 1.0).astype(np.float32)

            self._res_mean = torch.from_numpy(res_mean).view(1, num_y, 1, 1)
            self._res_std = torch.from_numpy(res_std).view(1, num_y, 1, 1)

            self.metadata_dict["residual_norm"] = {
                "mean": res_mean.tolist(),
                "std": res_std.tolist(),
            }
            log.info(
                "Residual standardization stats: mean=%s std=%s",
                res_mean.tolist(),
                res_std.tolist(),
            )

        return train_dataset, valid_dataset

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _marginal_prob(y, t, beta_min, beta_max):
        """
        Sub-VP SDE marginal distribution parameters (identical to trainer_cpmgem).

            B(t) = β_min·t + ½(β_max − β_min)·t²
            mean = exp(−½ B(t))          (applied to the clean signal)
            std  = 1 − exp(−B(t))        (sub-VP; not the VP √(1−e^{−B}))

        Returns
        -------
        mean : (B, 1, 1, 1)
        std  : (B, 1, 1, 1)
        """
        B_t = beta_min * t + 0.5 * (beta_max - beta_min) * t**2
        log_mean_coeff = -0.5 * B_t
        mean = torch.exp(log_mean_coeff)[:, None, None, None]
        std = (1.0 - torch.exp(-B_t))[:, None, None, None]
        return mean, std

    # ─────────────────────────────────────────────────────────────────────────
    def model_backprop(
        self,
        model,
        data,
        optimizer,
        loss_function,
        noise_params,  # forwarded from training_params.kwargs
        device,
        is_this_training=True,
        **kwargs,  # absorbs standardize_residual (handled in __init__) etc.
    ):
        """
        One forward / backward pass for CPMGEM-residual.

            1.  r, c_low, c_high = data        (residual + low-res + regressor mean)
            2.  normalize c_low (r and c_high are already in normalized space)
            3.  standardize r → ~unit variance (if enabled)
            4.  t ~ U(t_min, 1); sub-VP mean(t), std(t)
            5.  ε ~ N(0, I);  r_t = mean·r + std·ε
            6.  ε̂ = model(r_t, cond_low=c_low, cond_high=c_high, t·999)
            7.  loss = MseLoss(ε̂, ε)

        Mirrors trainer_resdiff's data handling (only c_low is normalized here;
        r and c_high were stored to the residuals zarr already normalized) and
        trainer_cpmgem's sub-VP forward process and t·999 PE convention.
        """
        r, c_low, c_high = data
        non_blocking = self.device_type == "cuda"

        r = r.to(device, non_blocking=non_blocking)
        if c_low is not None:
            c_low = c_low.to(device, non_blocking=non_blocking)
        if c_high is not None:
            c_high = c_high.to(device, non_blocking=non_blocking)

        # --- GPU-side normalization: only c_low (mirrors trainer_resdiff) ---
        if c_low is not None and self.norm_x is not None:
            c_low = self.norm_x(c_low)

        # --- Standardize the residual to ~unit variance for the sub-VP marginal ---
        if self.standardize_residual and self._res_mean is not None:
            if self._res_mean.device != r.device:
                self._res_mean = self._res_mean.to(r.device)
                self._res_std = self._res_std.to(r.device)
            r = (r - self._res_mean) / self._res_std

        beta_min = noise_params["beta_min"]
        beta_max = noise_params["beta_max"]
        t_min = noise_params.get("t_min", 1e-5)
        B = r.shape[0]

        # ── Sample continuous time t ∈ [t_min, 1] — kept in fp32 ──────────────
        t = torch.rand(B, device=device) * (1.0 - t_min) + t_min  # (B,)

        # ── Sub-VP forward diffusion on the residual — kept in fp32 ───────────
        mean, std = self._marginal_prob(r, t, beta_min, beta_max)
        eps = torch.randn_like(r)
        r_t = mean * r + std * eps

        optimizer.zero_grad(set_to_none=True)

        # ── Predict noise ε̂ + loss under AMP autocast ────────────────────────
        # The continuous sub-VP convention (score_sde_pytorch / mlde) feeds the
        # sinusoidal PE the scalar label t·999, not raw t. cond_high carries the
        # regressor mean ŷ_det (already at predictand resolution).
        t_label = t * 999.0
        with self._amp_ctx():
            eps_pred = model(
                x=r_t,
                t=t_label,
                cond_low=c_low,
                cond_high=c_high,
            )
            loss = loss_function(target=eps, output=eps_pred)

        if is_this_training:
            if self._scaler is not None:
                self._scaler.scale(loss).backward()
            else:
                loss.backward()

        return loss.detach()
