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
import torch.nn.functional as F
import xarray as xr

## Deep4production
from deep4production.core.downscalers.downscaler import downscaler
from deep4production.deep.utils import load_model
from deep4production.deep.models.diffusion.patching import (
    build_grid_patcher,
    assemble_cond_patches,
    run_regressor_patched,
)
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
        physical_bounds=None,
        unit_conversion=None,
    ):
        super().__init__(
            id_dir=id_dir,
            input_data=input_data,
            model_file=model_file,
            saving_info=saving_info,
            ensemble_size=ensemble_size,
            graph=graph,
            forcing_data=forcing_data,
            physical_bounds=physical_bounds,
            unit_conversion=unit_conversion,
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
        self.regressor, reg_meta = load_model(
            path=reg_path, map_location=self.device, return_metadata=True
        )
        self.regressor.to(self.device).eval()
        log.info("Regressor loaded from %s", reg_path)

        # ── Optional patched (tiled) inference ────────────────────────────────
        # Two independent patchers, each reconstructed from the geometry stored at
        # training: `diff_patcher` from THIS run's metadata (diffusion stage) and
        # `reg_patcher` from the REGRESSOR checkpoint's metadata (regression stage).
        # Either may be absent (whole-domain). Both tile the full (H_y, W_y) grid.
        self.diff_patcher = self.reg_patcher = None
        self.diff_K = self.reg_K = 0
        dcfg = self.metadata.get("patching")
        if dcfg and dcfg.get("enabled", False):
            if not self.transform_to_2D_y:
                raise ValueError("patched diffusion inference requires transform_to_2D_y.")
            self.diff_patcher, self.diff_K = build_grid_patcher(dcfg, (self.H_y, self.W_y))
            log.info("Patched diffusion: %d patches per sampler step.", self.diff_patcher.patch_num)
        rcfg = reg_meta.get("patching")
        if rcfg and rcfg.get("enabled", False):
            self.reg_patcher, self.reg_K = build_grid_patcher(rcfg, (self.H_y, self.W_y))
            log.info("Patched regressor: %d patches (tiled mean).", self.reg_patcher.patch_num)

        # ── Residual standardization (inverse of pydataset_resdiff) ─────────────
        # When the run was trained with standardize_residuals, the diffusion model
        # outputs residuals in ~unit-variance space; recover physical (normalized
        # [-1,1]) residuals via r = r_std * std + mean before adding the regressor
        # mean. Absent for legacy raw-residual runs -> no transform applied.
        self._res_mean = self._res_std = None
        rn = self.metadata.get("residual_norm", None)
        if rn is not None and rn.get("standardize", False):
            mean = torch.tensor(rn["mean"], dtype=torch.float32, device=self.device)
            std = torch.tensor(rn["std"], dtype=torch.float32, device=self.device)
            # Broadcast over (B, C, H, W).
            self._res_mean = mean.view(1, -1, 1, 1)
            self._res_std = std.view(1, -1, 1, 1)
            log.info("Residual standardization active: de-standardizing samples.")

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
        if self.diff_patcher is not None:
            return self._edm_heun_sample_patched(model, c_low, c_high, shape)
        return self._edm_heun_sample(model, c_low, c_high, shape)

    # ─────────────────────────────────────────────────────────────────────────
    def _denoise_patched(self, model, x_full, sigma_scalar, cond_high_p, pos_embd_p, B):
        """One patched denoiser call: patch the full-domain latent, denoise all
        patches with the fixed conditioning, and fuse back to the full domain
        (spec §5). `sigma_scalar` is the single noise level of this sampler step,
        broadcast to every patch."""
        x_patches = self.diff_patcher.extract(x_full)  # (P_num*B, C, P, P)
        sigma_pb = sigma_scalar.expand(self.diff_patcher.patch_num * B)
        D_patches = model(
            x_patches, sigma_pb, cond_low=None,
            cond_high=cond_high_p, pos_embd=pos_embd_p,
        )
        return self.diff_patcher.fuse(D_patches)  # (B, C, H, W)

    # ─────────────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def _edm_heun_sample_patched(self, model, cond_low, cond_high, shape):
        """
        Patch-based EDM Heun sampler (spec §5). The conditioning patches and the
        global positional embedding are built ONCE (they do not change across
        steps); only the latent residual is re-patched → denoised → fused every
        step, so cross-patch coherence is enforced at each iteration while the
        churn noise is re-injected at full strength on the whole field.
        """
        sigmas = self._sigmas()
        B = shape[0]

        # Build the fixed per-patch conditioning once (spec §2): local crops of
        # [mean_hr | img_lr_hr] + global_lr thumbnail, plus the global PE.
        c_low_hr = F.interpolate(
            cond_low, size=(self.H_y, self.W_y), mode="bilinear", align_corners=False
        )
        cond_local = torch.cat([cond_high, c_low_hr], dim=1)  # [mean_hr | img_lr_hr]
        self.diff_patcher.new_origins(self.H_y, self.W_y, self.device)  # no-op (grid)
        cond_high_p, pos_embd_p = assemble_cond_patches(
            self.diff_patcher, cond_local, c_low_hr, self.diff_K
        )

        x = torch.randn(shape, device=self.device) * sigmas[0]
        for i in range(self.num_steps):
            sigma_cur, sigma_next = sigmas[i], sigmas[i + 1]
            gamma = (
                min(self.S_churn / self.num_steps, 2**0.5 - 1)
                if self.S_min <= sigma_cur <= self.S_max
                else 0.0
            )
            sigma_hat = sigma_cur * (1.0 + gamma)
            if gamma > 0:
                x = x + (sigma_hat**2 - sigma_cur**2).sqrt() * self.S_noise * torch.randn_like(x)

            denoised = self._denoise_patched(model, x, sigma_hat, cond_high_p, pos_embd_p, B)
            d_cur = (x - denoised) / sigma_hat
            x_next = x + (sigma_next - sigma_hat) * d_cur

            if sigma_next > 0:
                denoised_next = self._denoise_patched(
                    model, x_next, sigma_next, cond_high_p, pos_embd_p, B
                )
                d_next = (x_next - denoised_next) / sigma_next
                x_next = x + (sigma_next - sigma_hat) * 0.5 * (d_cur + d_next)

            x = x_next

        return x

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

            # ── High-res forcing (orography) fed to the REGRESSOR as cond_high ──
            # Distinct from `c_high` below, which is the regressor's MEAN output
            # (the diffusion model's cond_high). Preprocessed per date batch and
            # normalized with norm_f, mirroring pydataset_resdiff's residual build.
            # None when the regressor was trained without forcings.
            reg_cond_high = None
            if self.forcing_data is not None:
                reg_cond_high = self._stack_to_device(
                    [self._preprocess_forcing_date(d) for d in batch_dates]
                )  # (B, C_f, H_y, W_y)
                if self.norm_f is not None:
                    reg_cond_high = self.norm_f(reg_cond_high)

            # ── Deterministic mean (shared across members) ───────────────────
            # The regressor is a SongUNet used deterministically: x is always
            # zeros (no noisy input), t is always zeros (no noise level), and
            # the actual low-res conditioning enters via cond_low (plus the
            # optional high-res forcing reg_cond_high, e.g. orography).
            with torch.inference_mode(), self._amp_ctx():
                B_cur = c_low.shape[0]
                C_y = len(self.vars_y)
                if self.reg_patcher is not None:
                    # Tiled regressor mean: upsample cond_low to HR, then
                    # apply→forward→fuse over the regressor's own patch grid.
                    c_low_hr = F.interpolate(
                        c_low, size=(self.H_y, self.W_y),
                        mode="bilinear", align_corners=False,
                    )
                    c_high = run_regressor_patched(
                        self.regressor, c_low_hr, reg_cond_high,
                        self.reg_patcher, self.reg_K, C_y,
                    )  # (B, C_y, H_y, W_y)
                else:
                    x_dummy = torch.zeros(
                        B_cur, C_y, self.H_y, self.W_y, device=self.device
                    )
                    t_dummy = torch.zeros(B_cur, device=self.device)
                    c_high = self.regressor(
                        x=x_dummy, t=t_dummy, cond_low=c_low, cond_high=reg_cond_high
                    )  # (B, C_y, H_y, W_y)

            # ── Stochastic residual: one draw per member ─────────────────────
            for member in range(M):
                with self._amp_ctx():
                    r_hat = self.sample(
                        c_low=c_low, c_high=c_high, model=model
                    )  # (B, C_y, H_y, W_y)
                # Invert the per-channel residual standardization applied at
                # training (no-op for legacy raw-residual runs). c_high (regressor
                # mean) is already in normalized [-1,1] space, so the recovered
                # residual must be too before they are summed.
                if self._res_std is not None:
                    r_hat = r_hat * self._res_std + self._res_mean
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
        ds_out = self._stamp_units(ds_out)
        log.debug("Writing prediction xarray to %s\n%s", self.output_path, ds_out)
        ds_out.to_netcdf(self.output_path)
        self._log_clip_stats()
