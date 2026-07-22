"""
Trainer for the CPMGEM diffusion model (Addison et al. 2024, arXiv:2407.14158).

Implements the sub-VP SDE forward diffusion process in **continuous time**,
following Song et al. (2021) and the mlde repository
(https://github.com/henryaddison/mlde).

Sub-VP SDE marginal distribution at time t ∈ (0, 1]:
    B(t)    = β_min·t + ½(β_max − β_min)·t²
    mean(t) = exp(−½ B(t)) · y₀
    std(t)  = 1 − exp(−B(t))                 ← linear, NOT √. The square-root
                                               form is the VP marginal; sub-VP
                                               (Song 2021 Appx. B / yang-song's
                                               score_sde_pytorch subVPSDE) has
                                               variance (1−e^{−B})².
    y_t     = mean(t) + std(t) · ε,   ε ~ N(0, I)

The model predicts the noise ε; the training objective is
    L = E[ ‖ε̂(y_t, x_cond, t) − ε‖² ]
supervised with MseLoss (equivalent to the score-matching objective used in
the paper up to a constant weighting).

The standard deep4production pydataset is used — no residual pre-computation.

Authors:
    Jorge Baño-Medina
"""

import torch
from deep4production.core.trainers.trainer import trainer
from deep4production.utils.log import get_logger

log = get_logger("trainer.cpmgem")


class trainer_custom(trainer):
    """
    Trainer for CPMGEM: direct-generation diffusion with sub-VP SDE.

    Inherits all dataset / dataloader / MLflow logic from the base trainer.
    Overrides model_backprop to inject the continuous-time diffusion process.

    Additional YAML keys expected under training_params.kwargs:
        noise_params:
            beta_min  : float  – β at t=0 (paper: 0.1)
            beta_max  : float  – β at t=1 (paper: 20.0)
            t_min     : float  – lower bound for continuous t (default 1e-5)
        warmup_steps: int (optional) – linear LR warm-up steps (paper: 5000)
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
        tracker=None,
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
            tracker=tracker,
            normalizer_info_x=normalizer_info_x,
            normalizer_info_y=normalizer_info_y,
            normalizer_info_f=normalizer_info_f,
            hardware=hardware,
        )

        # Store SDE params for metadata under training_params, mirroring the
        # nested structure used by trainer_resdiff so downscalers can resolve
        # noise_params from a single canonical path: metadata.training_params.noise_params.
        noise_params = model_info["training_params"]["kwargs"]["noise_params"]
        self.metadata_dict.setdefault("training_params", {})["noise_params"] = (
            noise_params
        )
        log.info("CPMGEM trainer ready (continuous-time sub-VP SDE)")

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _marginal_prob(y, t, beta_min, beta_max):
        """
        Sub-VP SDE marginal distribution parameters.

        Returns
        -------
        mean : (B, 1, 1, 1)  – mean coefficient (applied to y₀)
        std  : (B, 1, 1, 1)  – standard deviation of the noise
        """
        # Sub-VP marginal:
        #   B(t) = β_min·t + ½(β_max − β_min)·t²
        #   mean = exp(−½ B(t))
        #   std  = 1 − exp(−B(t))            (sub-VP; not the VP √(1−e^{−B}))
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
        **kwargs,
    ):
        """
        One forward / backward pass for CPMGEM.

        Continuous-time diffusion process:
            1.  t ~ U(t_min, 1)
            2.  Compute mean(t), std(t) from sub-VP SDE
            3.  Sample ε ~ N(0, I)
            4.  y_t = mean(t) · y  +  std(t) · ε
            5.  ε̂ = model(y_t, x_cond=x, t)
            6.  loss = MseLoss(ε̂, ε)

        Parameters
        ----------
        model         : CPMGEM instance
        data          : (x, y, f) from standard pydataset DataLoader
                        x : (B, C_x, H_x, W_x)  low-res conditioning
                        y : (B, C_y, H_y, W_y)  high-res target
                        f : (B, C_f, H_y, W_y)  high-res conditioning (e.g.
                            fine-scale orography), already at predictand
                            resolution; or the "N/A" sentinel when no forcings
                            are configured. When present it is routed to the
                            SongUNet ``cond_high`` stream and concatenated with
                            the target grid *after* cond_low is upsampled —
                            requires model_params.cond_high_channels == C_f.
        optimizer     : torch.optim.Optimizer
        loss_function : MseLoss(target=ε, output=ε̂)
        noise_params  : dict with beta_min, beta_max, t_min
        device        : str
        is_this_training : bool
        """
        x, y, f = data
        non_blocking = self.device_type == "cuda"
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        B = y.shape[0]

        # High-res conditioning (cond_high). pydataset emits the "N/A" string
        # sentinel when no forcings are configured; only a real tensor is routed
        # to the model. This keeps recipes without forcings (cond_high_channels=0)
        # working unchanged, where cond_high stays None.
        # NOTE: test for a tensor rather than for the sentinel -- the DataLoader's
        # default collate turns the per-sample "N/A" strings into a LIST of B
        # strings, so ``isinstance(f, str)`` is False for a batch and the code
        # would fall through to ``f.to(device)`` on a list.
        use_cond_high = torch.is_tensor(f)
        if use_cond_high:
            f = f.to(device, non_blocking=non_blocking)

        # --- GPU-side normalization (predictors + predictands + forcings) ---
        # Predictand normalization is essential for cpmgem because the diffusion
        # process operates on the normalized field y (e.g. sqrt(pr) rescaled to
        # [-1, 1]). The operator (sqrt) is still applied on CPU per-sample by
        # pydataset; only the affine rescale happens here. The forcing f (e.g.
        # orography) is normalized with its own InputNormalizer (norm_f) when a
        # forcing normalizer is configured.
        if use_cond_high:
            x, y, f = self._normalize_inputs(x=x, y=y, f=f)
        else:
            x, y, _ = self._normalize_inputs(x=x, y=y)

        beta_min = noise_params["beta_min"]
        beta_max = noise_params["beta_max"]
        t_min = noise_params.get("t_min", 1e-5)

        # ── Sample continuous time t ∈ [t_min, 1] — kept in fp32 ─────────────
        t = torch.rand(B, device=device) * (1.0 - t_min) + t_min  # (B,)

        # ── Sub-VP SDE marginal distribution ─────────────────────────────────
        mean, std = self._marginal_prob(y, t, beta_min, beta_max)

        # ── Forward diffusion — kept in fp32 ──────────────────────────────────
        eps = torch.randn_like(y)
        y_t = mean * y + std * eps

        optimizer.zero_grad(set_to_none=True)

        # ── Predict noise ε̂ + loss under AMP autocast ────────────────────────
        # SongUNet signature: (x, t, cond_low, cond_high). cond_low is the
        # low-res predictor stream (upsampled inside the UNet); cond_high carries
        # any high-res forcing already at predictand resolution (e.g. orography)
        # and is None when no forcings are configured.
        #
        # Continuous sub-VP convention (score_sde_pytorch / mlde): the scalar
        # noise label passed to the sinusoidal PE is `t · 999`, not raw t.
        # The PE uses frequencies in [1e-4, 1]; feeding raw t∈(0,1] collapses
        # most channels to ≈1 (cos) / ≈0 (sin) and loses time resolution.
        # The UNet itself is agnostic — it consumes whatever scalar label the
        # trainer provides — so the rescaling lives here.
        t_label = t * 999.0
        with self._amp_ctx():
            eps_pred = model(
                x=y_t,
                t=t_label,
                cond_low=x,
                cond_high=f if use_cond_high else None,
            )
            loss = loss_function(target=eps, output=eps_pred)

        if is_this_training:
            if self._scaler is not None:
                self._scaler.scale(loss).backward()
            else:
                loss.backward()

        return loss.detach()
