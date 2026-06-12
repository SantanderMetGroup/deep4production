"""
Downscaler for ResDiff / CorrDiff (Mardani et al. 2023).

Two-stage prediction: a deterministic regressor produces the mean prediction
y_hat = regressor(x); a diffusion U-Net (EDM-preconditioned SongUNet) samples
the stochastic residual r_hat; the final downscaled field is

    y = y_hat + r_hat          (in the normalised / operator-transformed space)

The residual is sampled with the EDM Heun sampler (Karras et al. 2022, Alg. 1/2),
feeding the preconditioner at every step. Stochasticity can be controlled with
the S_churn / S_min / S_max / S_noise parameters; setting S_churn=0 yields the
deterministic Heun sampler (recommended baseline).

This downscaler inherits all pre/post-processing (normalisation, operator
inversion, xarray formatting) from the base `downscaler` class. It overrides
only the prediction step.

Extra constructor argument (forwarded via ``d4p_downscaler.kwargs`` in the
inference YAML):

    sampling_params (dict, optional):
        num_steps (int)  – number of Heun steps                     (default: 18)
        sigma_min (float) – min noise level                          (from metadata)
        sigma_max (float) – max noise level                          (from metadata)
        rho       (float) – schedule curvature (EDM rho)             (default: 7.0)
        S_churn   (float) – stochastic churn amplitude               (default: 0.0)
        S_min     (float) – churn lower noise bound                  (default: 0.0)
        S_max     (float) – churn upper noise bound                  (default: inf)
        S_noise   (float) – churn noise scale                        (default: 1.0)

    path_regressor (str, optional):
        Override the regressor checkpoint path stored in training metadata.
        Useful if the regressor file has moved since training.

Author:
    Jorge Baño-Medina
"""

import numpy as np
import torch
import xarray as xr

## Deep4production
from deep4production.core.downscalers.downscaler import downscaler
from deep4production.deep.utils import load_model
from deep4production.utils.trans import from_pred_to_xarray
from deep4production.utils.log import get_logger

log = get_logger("downscaler.resdiff")


class downscaler_custom(downscaler):
    """
    ResDiff / CorrDiff downscaler.

    Overrides `downscale` to:
      1. Preprocess x into normalised low-res conditioning `c_low`.
      2. Run the deterministic regressor to produce `y_hat` (high-res mean).
      3. Sample the residual `r_hat` with the EDM Heun sampler, conditioned
         on `(c_low, y_hat)`.
      4. Return `y_hat + r_hat` (still in normalised space; base downscaler's
         `postprocess` denormalises).
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
        path_regressor=None,
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

        # ── Sampler parameters ────────────────────────────────────────────────
        # Noise range comes from training metadata by default; YAML can override.
        meta_noise = self.metadata.get("training_params", {}).get("noise_params", {})
        sp = sampling_params or {}

        self.num_steps = int(sp.get("num_steps", 18))
        self.sigma_min = float(sp.get("sigma_min", meta_noise.get("sigma_min", 0.002)))
        self.sigma_max = float(sp.get("sigma_max", meta_noise.get("sigma_max", 80.0)))
        self.rho = float(sp.get("rho", 7.0))
        self.S_churn = float(sp.get("S_churn", 0.0))
        self.S_min = float(sp.get("S_min", 0.0))
        self.S_max = float(sp.get("S_max", float("inf")))
        self.S_noise = float(sp.get("S_noise", 1.0))

        # ── Regressor (the deterministic mean-prediction network) ──────────────
        # Path priority: YAML override  >  training metadata  >  error.
        reg_path = path_regressor or self.metadata.get("path_regressor", None)
        if reg_path is None:
            raise ValueError(
                "ResDiff downscaler needs a regressor checkpoint: provide "
                "`path_regressor` in the YAML or ensure it was saved in the "
                "training metadata by trainer_resdiff."
            )
        self.regressor = load_model(path=reg_path, map_location=self.device)
        self.regressor.to(self.device).eval()
        log.info("Regressor loaded from %s", reg_path)

        log.info(
            "ResDiff sampler: steps=%d sigma=[%g, %g] rho=%g S_churn=%g",
            self.num_steps,
            self.sigma_min,
            self.sigma_max,
            self.rho,
            self.S_churn,
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _sigmas(self) -> torch.Tensor:
        """EDM noise schedule (Karras et al. 2022, Eq. 5):
            σ_i = (σ_max^{1/ρ} + i/(N-1) · (σ_min^{1/ρ} − σ_max^{1/ρ}))^ρ
        with an explicit σ_N = 0 appended for the final clean step.
        """
        N = self.num_steps
        ramp = torch.linspace(0, 1, N, device=self.device)
        inv_rho = 1.0 / self.rho
        sigmas = (
            self.sigma_max**inv_rho
            + ramp * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho
        return torch.cat([sigmas, torch.zeros(1, device=self.device)])  # (N+1,)

    # ─────────────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def _edm_heun_sample(
        self,
        precond,
        cond_low: torch.Tensor,
        cond_high: torch.Tensor,
        shape: tuple,
    ) -> torch.Tensor:
        """
        EDM Heun sampler (Karras et al. 2022, Alg. 1 deterministic / Alg. 2
        with churn for stochastic). Returns the sample at σ=0.

        At every step we call the preconditioner, which returns D_θ directly
        (noise-variance-weighted skip connections already applied), so we can
        treat the denoiser output as the clean estimate.
        """
        sigmas = self._sigmas()
        B = shape[0]

        # x ~ N(0, σ_max² I)
        x = torch.randn(shape, device=self.device) * sigmas[0]

        for i in range(self.num_steps):
            sigma_cur = sigmas[i]
            sigma_next = sigmas[i + 1]

            # Stochastic churn: σ_cur → σ_hat = σ_cur·(1 + γ)
            gamma = (
                min(self.S_churn / self.num_steps, 2**0.5 - 1)
                if self.S_min <= sigma_cur <= self.S_max
                else 0.0
            )
            sigma_hat = sigma_cur * (1.0 + gamma)
            if gamma > 0:
                x = x + (
                    sigma_hat**2 - sigma_cur**2
                ).sqrt() * self.S_noise * torch.randn_like(x)

            # Euler step: d = (x − D(x, σ_hat)) / σ_hat
            sigma_b = sigma_hat.expand(B)
            denoised = precond(x, sigma_b, cond_low=cond_low, cond_high=cond_high)
            d_cur = (x - denoised) / sigma_hat
            x_next = x + (sigma_next - sigma_hat) * d_cur

            # Heun 2nd-order correction (skip on the clean step σ_next = 0)
            if sigma_next > 0:
                sigma_nb = sigma_next.expand(B)
                denoised_next = precond(
                    x_next, sigma_nb, cond_low=cond_low, cond_high=cond_high
                )
                d_next = (x_next - denoised_next) / sigma_next
                x_next = x + (sigma_next - sigma_hat) * 0.5 * (d_cur + d_next)

            x = x_next

        return x

    # ─────────────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def sample(self, c_low: torch.Tensor, c_high: torch.Tensor, model) -> torch.Tensor:
        """Draw residual samples r_hat for a batch of dates given (c_low, c_high)."""
        C_y = len(self.vars_y)
        B = c_low.shape[0]
        shape = (B, C_y, self.H_y, self.W_y)
        return self._edm_heun_sample(model, c_low, c_high, shape)

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
        Override base `downscale`: date-outer, member-inner.

        Per date batch:
          1. Preprocess c_low ONCE.
          2. Run regressor ONCE  →  c_high  (deterministic, shared across members).
          3. For each member: draw fresh residual r_hat, accumulate (c_high + r_hat).

        This avoids the previous structure where the regressor was rerun for
        every ensemble member even though its output is identical.
        """
        if verbose:
            log.info("Starting ResDiff downscaling process")

        if model is None:
            model = self.model
        self._amp_dtype = self._parse_amp_dtype(amp_dtype)
        model = self._maybe_compile(model, compile)
        # Optionally compile the regressor too (separate _is_compiled flag would be ideal,
        # but practically the regressor is a small one-shot pass — skip compile for it).
        model.eval()

        all_dates_np = [np.datetime64(d) for d in self.target_dates]
        T = len(self.target_dates)
        M = self.ensemble_size
        n_batches = (T + batch_size - 1) // batch_size

        # Per-member numpy buffers (each will end up as (T, C, G) once concatenated).
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
            c_low = self._stack_to_device(
                [self._preprocess_single_date(d) for d in batch_dates]
            )  # (B, C_x, H_x, W_x)

            # ── GPU-side input normalization (mirrors trainer + residuals build) ──
            # The regressor and the EDM model were both trained with normalized
            # cond_low (see pydataset_resdiff and trainer_resdiff).
            if self.norm_x is not None:
                c_low = self.norm_x(c_low)

            # ── Deterministic mean (shared across members) ───────────────────
            # The regressor is a SongUNet used deterministically: x is always
            # zeros (no noisy input), t is always zeros (no noise level), and
            # the actual low-res conditioning enters via cond_low.
            with torch.inference_mode(), self._amp_ctx():
                B_cur = c_low.shape[0]
                x_dummy = torch.zeros(
                    B_cur, len(self.vars_y), self.H_y, self.W_y, device=self.device
                )
                t_dummy = torch.zeros(B_cur, device=self.device)
                c_high = self.regressor(
                    x=x_dummy, t=t_dummy, cond_low=c_low
                )  # (B, C_y, H_y, W_y)

            # ── High-res forcings (cond_high = [y_hat, f]) ───────────────────
            # Concatenate any configured forcings (e.g. orography) onto the
            # regressor mean, in the SAME order as pydataset_resdiff at training
            # (regressor mean first, forcing second). The base downscaler loads
            # forcing_data and builds norm_f from metadata.
            if self.forcing_data is not None:
                f_cond = self._stack_to_device(
                    [self._preprocess_forcing_date(d) for d in batch_dates]
                )  # (B, C_f, H_y, W_y)
                if self.norm_f is not None:
                    f_cond = self.norm_f(f_cond)
                c_high = torch.cat([c_high, f_cond], dim=1)

            # ── Stochastic residual: one draw per member ─────────────────────
            for member in range(M):
                with self._amp_ctx():
                    r_hat = self.sample(
                        c_low=c_low, c_high=c_high, model=model
                    )  # (B, C_y, H_y, W_y)
                # Combined prediction in normalised y-space → denormalize on GPU
                # so it arrives in operator-applied space, which is what
                # _postprocess_numpy expects (operator inverse runs on CPU).
                p_gpu = c_high + r_hat
                if self.norm_y is not None:
                    p_gpu = self.norm_y.inverse_transform(p_gpu.float())
                p_cpu = self._async_d2h(p_gpu.float())
                if self._cuda:
                    torch.cuda.synchronize()
                member_buffers[member].append(self._postprocess_numpy(p_cpu.numpy()))
                del r_hat, p_gpu, p_cpu

            del c_low, c_high

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
        log.debug("Writing prediction xarray to %s\n%s", self.output_path, ds_out)
        ds_out.to_netcdf(self.output_path)
