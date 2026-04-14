"""
Trainer for the CPMGEM diffusion model (Addison et al., 2024, arXiv:2407.14158).

Implements the sub-VP SDE forward diffusion process (discrete, DDPM-style):
  β_t   = linspace(β_min, β_max, T)
  ᾱ_t  = ∏_{s=1}^{t} (1 − β_s)
  y_t   = √ᾱ_t · y₀  +  √(1 − ᾱ_t) · ε,    ε ~ N(0, I)

The model is trained to predict ε from (y_t, x_cond, t), supervised with
MseLoss.  This is a direct-generation diffusion model — it learns the full
high-resolution output, NOT a residual on top of a deterministic regressor
(contrast with trainer_resdiff).

The standard pydataset is used (no residual pre-computation needed).

Authors:
    Jorge Baño-Medina
"""

import torch
from deep4production.core.trainers.trainer import trainer


class trainer_custom(trainer):
    """
    Trainer for CPMGEM: direct-generation diffusion model with sub-VP SDE.

    Inherits all dataset / dataloader / MLflow logic from the base trainer.
    Only overrides model_backprop to inject the diffusion forward process.

    Additional YAML keys expected under training_params.kwargs:
        noise_params:
            T        : int   – number of diffusion steps (e.g. 1000)
            beta_min : float – smallest noise level (e.g. 0.0001)
            beta_max : float – largest  noise level (e.g. 0.02)
    """

    def __init__(self, data, dataloader, id_dir, model_info, graph, d4dpy, Mlflow):
        super().__init__(
            data=data,
            dataloader=dataloader,
            id_dir=id_dir,
            model_info=model_info,
            graph=graph,
            d4dpy=d4dpy,
            Mlflow=Mlflow,
        )

        # ── Pre-compute sub-VP SDE schedule ──────────────────────────────────
        noise_params = model_info["training_params"]["kwargs"]["noise_params"]
        T        = noise_params["T"]
        beta_min = noise_params["beta_min"]
        beta_max = noise_params["beta_max"]

        betas      = torch.linspace(beta_min, beta_max, T, dtype=torch.float64)
        alphas_bar = torch.cumprod(1.0 - betas, dim=0).float()

        # Store on CPU; moved to device on each forward call
        self.T                            = T
        self.sqrt_alphas_bar              = alphas_bar.sqrt()           # (T,)
        self.sqrt_one_minus_alphas_bar    = (1.0 - alphas_bar).sqrt()  # (T,)

        # ── Persist noise schedule in metadata for reproducibility ────────────
        self.metadata_dict["noise_params"] = noise_params
        print("📦 CPMGEM TRAINER READY")

    # ─────────────────────────────────────────────────────────────────────────
    def model_backprop(
        self,
        model,
        data,
        optimizer,
        loss_function,
        device,
        noise_params,          # passed via training_params.kwargs (see YAML)
        is_this_training=True,
        **kwargs,
    ):
        """
        One forward/backward pass for CPMGEM.

        Diffusion forward process:
            1. Sample t ~ Uniform{0, …, T-1}
            2. Sample ε ~ N(0, I)
            3. Compute y_t = √ᾱ_t · y  +  √(1 − ᾱ_t) · ε
            4. Predict ε̂ = model(y_t, x_cond=x, t_norm)
            5. loss = MseLoss(ε̂, ε)

        Parameters
        ----------
        model : nn.Module
            CPMGEM instance.
        data : tuple
            (x, y, f) from the standard pydataset DataLoader.
            x : (B, C_x, H_x, W_x) – low-res predictors
            y : (B, C_y, H_y, W_y) – high-res target
            f : forcing tensor or "N/A" sentinel (unused here)
        optimizer : torch.optim.Optimizer
        loss_function : callable   MseLoss(target=ε, output=ε̂)
        device : str
        noise_params : dict        Forwarded from training_params.kwargs
        is_this_training : bool
        """
        x, y, _ = data          # forcings are not used in the diffusion step
        x = x.to(device)        # (B, C_x, H_x, W_x)  low-res conditioning
        y = y.to(device)        # (B, C_y, H_y, W_y)  high-res target
        B = y.shape[0]

        # ── Sample random timesteps t ∈ {0, …, T-1} ──────────────────────────
        t_idx = torch.randint(0, self.T, (B,))   # (B,)  0-indexed

        # ── Look up schedule coefficients ─────────────────────────────────────
        sqrt_ab  = self.sqrt_alphas_bar[t_idx].view(B, 1, 1, 1).to(device)
        sqrt_1mab = self.sqrt_one_minus_alphas_bar[t_idx].view(B, 1, 1, 1).to(device)

        # ── Forward diffusion: y_t = √ᾱ_t · y + √(1 − ᾱ_t) · ε ──────────────
        eps  = torch.randn_like(y)
        y_t  = sqrt_ab * y + sqrt_1mab * eps

        # ── Normalise t to [0, 1] for the sinusoidal embedding ────────────────
        t_norm = ((t_idx.float() + 1.0) / self.T).to(device)   # (B,)

        # ── Model forward: predict ε ──────────────────────────────────────────
        optimizer.zero_grad()
        eps_pred = model(y_t=y_t, x_cond=x, t=t_norm)

        # ── Loss ──────────────────────────────────────────────────────────────
        loss = loss_function(target=eps, output=eps_pred)

        if is_this_training:
            loss.backward()

        return loss.item()
