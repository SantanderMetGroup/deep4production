"""
Downscaler for CPMGEM: diffusion-based climate downscaling via reverse sub-VP SDE.

Implements the Euler-Maruyama reverse-SDE sampler that is the exact time-reversal
of the forward process used in trainer_cpmgem.  Sub-VP SDE (Song et al. 2021,
Appx. B / Addison et al. 2024, arXiv:2407.14158 / mlde):

    B(t)    = β_min·t + ½(β_max − β_min)·t²
    f(y, t) = −½ β(t) y                              (forward drift)
    g²(t)   = β(t)·(1 − e^{−2B(t)})                  (forward diffusion², sub-VP)
    σ(t)    = 1 − e^{−B(t)}                           (marginal std, sub-VP)

The reverse SDE drift is f − g²·∇log p with score = −ε̂/σ, giving

    drift_rev = −½β y + g²·ε̂/σ
              = −½β y + β·(1 + e^{−B})·ε̂                (using 1−e^{−2B} = (1−e^{−B})(1+e^{−B}))

Reverse Euler-Maruyama step (time decreasing, step size h > 0; equivalent to
yang-song's EulerMaruyamaPredictor with dt = −1/N):

    y_{t-h} = y_t  +  h · [+½β(t) y_t  −  β(t)(1 + e^{−B(t)}) ε̂]  +  √(g²(t)·h) · z

where ε̂ = CPMGEM(y_t, x_cond, t · 999),  z ∼ N(0, I).
The 999 factor matches the score_sde_pytorch / mlde convention for continuous
sub-VP positional embeddings (the network's noise label is rescaled to span the
[0, 999] grid the sinusoidal PE was designed for) and must agree with the
rescaling applied at training time in trainer_cpmgem.

Authors:
    Jorge Baño-Medina
"""

import numpy as np
import torch
import xarray as xr

## Deep4production
from deep4production.core.downscalers.downscaler import downscaler
from deep4production.utils.trans import from_pred_to_xarray
from deep4production.utils.log import get_logger

log = get_logger("downscaler.cpmgem")


##################################################################################################################################
class downscaler_custom(downscaler):
    """
    CPMGEM downscaler: replaces the single deterministic forward pass of the base
    class with a full reverse-diffusion sampling chain.

    Inherits all dataset / preprocessing / postprocessing logic from the base
    downscaler.  Only ``downscale`` and the sampling helpers are new.

    Extra constructor argument (forwarded via ``d4p_downscaler.kwargs`` in the
    inference YAML):

        sampling_params (dict, optional):
            num_steps (int)   – number of reverse SDE steps       (default: 500)
            beta_min  (float) – β at t=0; must match training      (from metadata)
            beta_max  (float) – β at t=1; must match training      (from metadata)
            t_min     (float) – lower bound for t at sampling      (default: 1e-3)
            denoise   (bool)  – noise-free final step for sharper output (default: True)

    Notes
    -----
    ``beta_min`` and ``beta_max`` are read automatically from the checkpoint
    metadata saved by ``trainer_cpmgem``; override only if you know what you
    are doing. ``t_min`` is *decoupled* from training: at training time the
    SDE is sampled down to 1e-5, but the score correction g²·ε̂/σ blows up as
    σ → 0, so reverse sampling stops early at 1e-3 (mlde / score_sde_pytorch
    default). Override via ``sampling_params.t_min`` if needed.
    """

    def __init__(
        self,
        id_dir,
        input_data,
        model_file=None,
        saving_info=None,
        ensemble_size=1,
        graph=None,
        forcing_data=None,
        sampling_params=None,
    ):
        super().__init__(
            id_dir=id_dir,
            input_data=input_data,
            model_file=model_file,
            saving_info=saving_info,
            ensemble_size=ensemble_size,
            graph=graph,
            forcing_data=forcing_data,
        )

        # ── SDE / sampler parameters ──────────────────────────────────────────
        # beta_min/beta_max are read from training metadata (must match) but
        # t_min is *decoupled*: training samples t ∈ [1e-5, 1] for coverage,
        # while reverse sampling stops at 1e-3 to avoid the σ → 0 blow-up of
        # the score correction g²·ε̂/σ (mlde / score_sde_pytorch convention).
        # YAML sampling_params can override any of these.
        meta_noise = self.metadata.get("training_params", {}).get("noise_params", {})
        sp = sampling_params or {}

        self.beta_min = float(sp.get("beta_min", meta_noise.get("beta_min", 0.1)))
        self.beta_max = float(sp.get("beta_max", meta_noise.get("beta_max", 20.0)))
        self.t_min = float(sp.get("t_min", 1e-3))
        self.num_steps = int(sp.get("num_steps", 500))
        self.denoise = bool(sp.get("denoise", True))

        log.info(
            "CPMGEM sampler: beta_min=%g beta_max=%g t_min=%g steps=%d denoise=%s",
            self.beta_min,
            self.beta_max,
            self.t_min,
            self.num_steps,
            self.denoise,
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _sde_coeffs(self, t: torch.Tensor):
        """
        Sub-VP SDE coefficients at a scalar time t.

        Returns
        -------
        beta_t : scalar – β(t)
        g2     : scalar – g²(t) = β(t)(1 − e^{−2B(t)})            (sub-VP)
        std_t  : scalar – σ(t)  = 1 − e^{−B(t)}                   (sub-VP)
        """
        beta_t = self.beta_min + (self.beta_max - self.beta_min) * t
        B_t = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
        exp_neg_B = torch.exp(-B_t)
        std_t = (1.0 - exp_neg_B).clamp(min=1e-10)
        g2 = beta_t * (1.0 - exp_neg_B * exp_neg_B)
        return beta_t, g2, std_t

    # ─────────────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def _reverse_step(
        self,
        y_t: torch.Tensor,
        x_cond: torch.Tensor,
        t_scalar: torch.Tensor,
        dt: float,
        model,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """
        One Euler-Maruyama reverse-SDE step: t → t − dt.

        Parameters
        ----------
        y_t      : (B, C_y, H_y, W_y)  noisy sample at time t
        x_cond   : (B, C_x, H_x, W_x)  low-res conditioning (constant across steps)
        t_scalar : scalar tensor         current time
        dt       : float                 step size (positive)
        model    : CPMGEM instance
        add_noise: bool                  whether to add stochastic noise term

        Returns
        -------
        y_{t-dt} : (B, C_y, H_y, W_y)
        """
        B = y_t.shape[0]
        t_batch = t_scalar.expand(B)  # (B,)

        # Predict noise  ε̂ = model(y_t, cond_low=x_cond, t · 999)
        # SongUNet signature: (x, t, cond_low, cond_high). The 999 factor
        # rescales the continuous noise label to the integer-positional-encoding
        # grid the network was trained on (mlde / score_sde_pytorch convention,
        # applied identically in trainer_cpmgem).
        t_label = t_batch * 999.0
        eps_pred = model(x=y_t, t=t_label, cond_low=x_cond)

        # SDE coefficients at current t
        beta_t, g2, std_t = self._sde_coeffs(t_scalar)

        # Reverse Euler-Maruyama step (yang-song convention with dt < 0,
        # rewritten with positive `dt` here as a forward step in reverse time):
        #   y_{t-h} = y_t + h · (½β y_t − g²·ε̂/σ) + √(g²·h)·z
        # The +½β·y_t term grows |y| back from noise toward data; the
        # −g²·ε̂/σ term applies the score correction (= β·(1+e^{−B})·ε̂ for
        # sub-VP after using 1−e^{−2B} = (1−e^{−B})(1+e^{−B})).
        y_next = y_t + dt * (0.5 * beta_t * y_t - g2 * (eps_pred / std_t))

        if add_noise:
            noise_std = torch.sqrt((g2 * dt).clamp(min=0.0))
            y_next = y_next + noise_std * torch.randn_like(y_t)

        return y_next

    # ─────────────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def sample(self, x_cond: torch.Tensor, model) -> torch.Tensor:
        """
        Full reverse-diffusion chain: y_T ∼ N(0, I)  →  y_0.

        Parameters
        ----------
        x_cond : (B, C_x, H_x, W_x)  preprocessed low-res conditioning (batch of dates)
        model  : CPMGEM

        Returns
        -------
        (B, C_y, H_y, W_y)  — all B dates sampled in parallel on the GPU
        """
        B = x_cond.shape[0]
        C_y = len(self.vars_y)
        y_t = torch.randn(B, C_y, self.H_y, self.W_y, device=self.device)

        ts = torch.linspace(1.0, self.t_min, self.num_steps + 1, device=self.device)
        dt = float((1.0 - self.t_min) / self.num_steps)

        for i, t in enumerate(ts[:-1]):
            # Suppress noise on the very last step when denoise=True.
            # This "tweedie" denoising step reduces residual variance.
            last_step = i == self.num_steps - 1
            add_noise = not (last_step and self.denoise)
            y_t = self._reverse_step(y_t, x_cond, t, dt, model, add_noise=add_noise)

        log.debug(
            "y_0 min=%.3f max=%.3f mean=%.3f std=%.3f",
            float(y_t.min()),
            float(y_t.max()),
            float(y_t.mean()),
            float(y_t.std()),
        )
        return y_t

    # ─────────────────────────────────────────────────────────────────────────
    def downscale(
        self,
        model=None,
        return_pred=False,
        verbose=True,
        batch_size=1,
        amp_dtype=None,
        compile=False,
    ):
        """
        Override base ``downscale``: runs reverse-diffusion sampling.

        Loop structure: date-outer, member-inner.
          - For each date batch: preprocess the low-res conditioning ONCE.
          - Then for each ensemble member: run a fresh reverse-SDE chain
            with independent Gaussian noise.

        Parameters
        ----------
        batch_size : int
            Number of dates whose reverse-diffusion chains run in parallel on
            the GPU. All B chains share the same num_steps SDE iterations.
        amp_dtype  : 'bfloat16' / 'float16' / None
            Mixed-precision autocast for the score-network forward inside each
            SDE step. bf16 is the recommended default for diffusion sampling.
        compile    : bool
            If True, wraps the score network with torch.compile(dynamic=True,
            mode="reduce-overhead"). The 500 inner SDE steps amortise the
            one-shot compile cost very quickly.
        """
        if verbose:
            log.info("Starting CPMGEM downscaling process")

        if model is None:
            model = self.model
        self._amp_dtype = self._parse_amp_dtype(amp_dtype)
        model = self._maybe_compile(model, compile)
        model.eval()

        all_dates_np = [np.datetime64(d) for d in self.target_dates]
        T = len(self.target_dates)
        M = self.ensemble_size
        n_batches = (T + batch_size - 1) // batch_size

        # Per-member numpy buffers.
        member_buffers = [[] for _ in range(M)]

        for b_idx in range(n_batches):
            i = b_idx * batch_size
            batch_dates = self.target_dates[i : i + batch_size]
            if verbose:
                log.info(
                    "Batch %d/%d: %s → %s (%d dates) x %d member(s)",
                    b_idx + 1,
                    n_batches,
                    batch_dates[0],
                    batch_dates[-1],
                    len(batch_dates),
                    M,
                )

            # ── Preprocess once per date batch ───────────────────────────────
            inp = self._stack_to_device(
                [self._preprocess_single_date(d) for d in batch_dates]
            )  # (B, C_x, H_x, W_x)

            # ── GPU-side input normalization (mirrors trainer_cpmgem) ─────────
            # The diffusion UNet was trained on normalized predictors via
            # trainer_cpmgem._normalize_inputs(x=x, y=y); inference must apply
            # the same affine before the reverse-SDE chain.
            if self.norm_x is not None:
                inp = self.norm_x(inp)

            # ── Reverse diffusion: one chain per member ──────────────────────
            for member in range(M):
                with self._amp_ctx():
                    p_torch = self.sample(x_cond=inp, model=model)  # (B, C_y, H_y, W_y)
                # Sample is in normalized y-space (target was normalized at
                # training time); denormalize on GPU so it arrives in
                # operator-applied space, which is what _postprocess_numpy
                # expects (operator inverse runs on CPU).
                if self.norm_y is not None:
                    p_torch = self.norm_y.inverse_transform(p_torch.float())
                p_cpu = self._async_d2h(p_torch.float())
                if self._cuda:
                    torch.cuda.synchronize()
                member_buffers[member].append(self._postprocess_numpy(p_cpu.numpy()))
                del p_torch, p_cpu

            del inp

        # ── Build xarray per member, concat along member ─────────────────────
        ds_out = []
        for m, buf in enumerate(member_buffers):
            all_preds_np = np.concatenate(buf, axis=0)  # (T, C, G)
            ds_member = from_pred_to_xarray(
                all_preds_np,
                all_dates_np,
                self.vars_y,
                self.lats,
                self.lons,
                self.template,
                self.H_y,
                self.W_y,
                precomputed_mask=self._template_mask,
            )
            ds_member = ds_member.assign_coords({"member": m})
            ds_out.append(ds_member)
        ds_out = xr.concat(ds_out, dim="member")

        if self.format_output:
            ds_out = self.formatting_func(ds_out, **self.formatting_kwargs)
        if return_pred:
            return ds_out
        log.debug("Writing prediction xarray\n%s", ds_out)
        ds_out.to_netcdf(self.output_path)
