"""
Trainer for EDM-direct — the EDM × direct cell of the 2×2 study.

This is the EDM analogue of CPMGEM-direct: the same direct (full-field) generation
target, but with the **EDM/CorrDiff diffusion formulation** instead of the
continuous-time sub-VP SDE. It completes the 2×2 by decoupling diffusion
formulation from parameterization target:

                         | Direct (predict full y) | Residual to regressor
    ---------------------|-------------------------|----------------------
    Sub-VP (CPMGEM)      | CPMGEM (trainer_cpmgem) | CPMGEM-residual
    EDM    (CorrDiff)    | EDM-direct (THIS)       | ResDiff (trainer_resdiff)

Mechanics — a hybrid of the two parents:
  * data path from CPMGEM-direct (trainer_cpmgem): the **standard pydataset**
    serves (x, y, f); the trainer normalizes predictors x, the full target field
    y, and any forcing f on the GPU. There is no regressor and no residual
    precomputation (so the base trainer's get_pydatasets is used unchanged).
  * diffusion math from ResDiff (trainer_resdiff): EDM log-normal σ sampling,
    an EDM-preconditioned backbone (build_edm_model → EDMPrecond(SongUNet)) that
    returns the denoised D_θ, and the λ(σ)-weighted denoising score-matching loss.

Unlike sub-VP, EDM absorbs the data scale through its sigma_data preconditioning,
so the full normalized field needs **no standardization** (this is exactly the
scale-handling sub-VP lacks). Conditioning is cond_low (predictors), with optional
cond_high (forcings, e.g. orography) — identical wiring to CPMGEM-direct.

Authors:
    Jorge Baño-Medina
"""

import torch
from deep4production.core.trainers.trainer import trainer
from deep4production.utils.log import get_logger

log = get_logger("trainer.edm_direct")


class trainer_custom(trainer):
    """
    Trainer for EDM-direct: EDM-formulation diffusion generating the full field.

    Inherits all dataset / dataloader / MLflow logic — including the standard
    get_pydatasets (no residual pydataset) — from the base trainer. Overrides
    model_backprop to run the EDM forward process directly on the normalized
    target field.

    Additional YAML keys under training_params.kwargs:
        noise_params:
            P_mean    : float  – mean of ln(σ)        (EDM default −1.2)
            P_std     : float  – std of ln(σ)         (EDM default  1.2)
            sigma_min : float  – clamp lower bound     (EDM default 0.002)
            sigma_max : float  – clamp upper bound     (EDM default 80.0)

    The model is expected to be an EDM-preconditioned backbone built via
    deep4production.deep.models.diffusion.edm_precond.build_edm_model; sigma_data
    lives in the preconditioner buffer and must match loss_params.sigma_data.
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

        # Store EDM noise params under the canonical metadata path
        # (metadata.training_params.noise_params), shared with trainer_resdiff /
        # trainer_cpmgem, so the downscaler resolves sigma_min/sigma_max from one
        # place.
        noise_params = model_info["training_params"]["kwargs"]["noise_params"]
        self.metadata_dict.setdefault("training_params", {})["noise_params"] = (
            noise_params
        )
        log.info("EDM-direct trainer ready (EDM SDE, direct full-field generation)")

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def sigma(P_mean, P_std, sigma_min, sigma_max, batch_size):
        """
        EDM log-normal noise schedule (Karras et al. 2022): ln(σ) ~ N(P_mean,
        P_std²), clamped to [sigma_min, sigma_max]. Returns (B, 1, 1, 1).

        Identical to trainer_resdiff.sigma; duplicated here so EDM-direct does not
        inherit ResDiff's residual-pydataset wiring.
        """
        z = torch.randn(batch_size, 1, 1, 1)
        sigma_t = torch.exp(P_mean + P_std * z)
        sigma_t = sigma_t.clamp(min=sigma_min, max=sigma_max)
        return sigma_t

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
        **kwargs,
    ):
        """
        One forward / backward pass for EDM-direct.

            1.  x, y, f = data        (predictors, full target field, optional forcing)
            2.  normalize x, y, f on the GPU
            3.  σ ~ EDM log-normal;  z ~ N(0, I);  y_t = y + σ·z
            4.  D_θ = model(y_t, σ, cond_low=x, cond_high=f)   (EDM preconditioner)
            5.  loss = λ(σ)·‖D_θ − y‖²   (WeightedDenoisingScoreMatchingLoss)

        Mirrors trainer_cpmgem's data handling / forcing sentinel and
        trainer_resdiff's EDM forward + λ(σ) loss, but the clean signal is the
        full field y rather than a residual.
        """
        x, y, f = data
        non_blocking = self.device_type == "cuda"
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)

        # High-res conditioning (cond_high). The standard pydataset emits the
        # "N/A" string sentinel when no forcings are configured; only a real
        # tensor is routed to the model (cond_high stays None otherwise, so
        # recipes with cond_high_channels=0 work unchanged).
        use_cond_high = not isinstance(f, str)
        if use_cond_high:
            f = f.to(device, non_blocking=non_blocking)

        # --- GPU-side normalization (predictors + target field + forcings) ---
        if use_cond_high:
            x, y, f = self._normalize_inputs(x=x, y=y, f=f)
        else:
            x, y, _ = self._normalize_inputs(x=x, y=y)

        P_mean = noise_params["P_mean"]
        P_std = noise_params["P_std"]
        sigma_min = noise_params["sigma_min"]
        sigma_max = noise_params["sigma_max"]
        B = y.shape[0]

        # ── EDM forward process on the full field — kept in fp32 ──────────────
        sigma_t = self.sigma(
            P_mean, P_std, sigma_min=sigma_min, sigma_max=sigma_max, batch_size=B
        ).to(device)
        z = torch.randn_like(y)
        y_t = y + sigma_t * z

        optimizer.zero_grad(set_to_none=True)

        # ── Denoise + λ(σ)-weighted loss under AMP autocast ───────────────────
        # The EDM preconditioner returns D_θ directly (c_in/c_skip/c_out/c_noise
        # applied inside). cond_low carries the low-res predictors; cond_high
        # carries any high-res forcing (None when unused).
        with self._amp_ctx():
            D_theta = model(
                x=y_t,
                sigma=sigma_t,
                cond_low=x,
                cond_high=f if use_cond_high else None,
            )
            loss = loss_function(target=y, output=D_theta, sigma_t=sigma_t)

        if is_this_training:
            if self._scaler is not None:
                self._scaler.scale(loss).backward()
            else:
                loss.backward()

        return loss.detach()
