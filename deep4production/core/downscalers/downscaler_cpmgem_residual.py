"""
Downscaler for CPMGEM-residual — inference for the sub-VP × residual cell.

Two-stage prediction, like ResDiff, but the stochastic stage is the CPMGEM sub-VP
reverse SDE instead of the EDM Heun sampler:

    1. deterministic regressor  →  ŷ_det = regressor(x)          (normalized space)
    2. sub-VP reverse SDE        →  r_std (standardized residual sample)
    3. un-standardize            →  r̂ = r_std · σ_res + μ_res
    4. combine                   →  y = ŷ_det + r̂               (normalized space)

The reverse sub-VP chain (_sde_coeffs / _reverse_step / sample) is inherited from
downscaler_cpmgem unchanged; the regressor mean ŷ_det is routed into the network's
cond_high stream (f_cond), exactly as the trainer feeds c_high. The residual
standardization stats (μ_res, σ_res) are read from training metadata
(`residual_norm`, written by trainer_cpmgem_residual) so the inverse matches
training.

Author:
    Jorge Baño-Medina
"""

import numpy as np
import torch
import xarray as xr

from deep4production.core.downscalers.downscaler_cpmgem import (
    downscaler_custom as downscaler_cpmgem,
)
from deep4production.deep.utils import load_model
from deep4production.utils.trans import from_pred_to_xarray
from deep4production.utils.log import get_logger

log = get_logger("downscaler.cpmgem_residual")


class downscaler_custom(downscaler_cpmgem):
    """
    CPMGEM-residual downscaler.

    Inherits the reverse sub-VP sampler from downscaler_cpmgem and adds the
    deterministic regressor (loaded as in downscaler_resdiff) plus the residual
    un-standardization step.

    Extra constructor argument (via ``d4p_downscaler.kwargs`` in the inference YAML):

        path_regressor (str, optional):
            Override the regressor checkpoint path stored in training metadata.

    All ``sampling_params`` (num_steps, beta_min, beta_max, t_min, denoise) are as
    documented in downscaler_cpmgem.
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
            sampling_params=sampling_params,
        )

        # ── Regressor (deterministic mean-prediction network) ──────────────────
        # Path priority: YAML override > training metadata > error. Mirrors
        # downscaler_resdiff.
        reg_path = path_regressor or self.metadata.get("path_regressor", None)
        if reg_path is None:
            raise ValueError(
                "CPMGEM-residual downscaler needs a regressor checkpoint: provide "
                "`path_regressor` in the YAML or ensure it was saved in the training "
                "metadata by trainer_cpmgem_residual."
            )
        self.regressor = load_model(path=reg_path, map_location=self.device)
        self.regressor.to(self.device).eval()
        log.info("Regressor loaded from %s", reg_path)

        # ── Residual standardization stats (inverse of the trainer's transform) ─
        # Stored as (1, C_y, 1, 1) tensors; default to (0, 1) → no-op if absent
        # (e.g. a model trained with standardize_residual=false).
        C_y = len(self.vars_y)
        res_norm = self.metadata.get("residual_norm", None)
        if res_norm is not None:
            mean = torch.tensor(res_norm["mean"], dtype=torch.float32)
            std = torch.tensor(res_norm["std"], dtype=torch.float32)
        else:
            mean = torch.zeros(C_y, dtype=torch.float32)
            std = torch.ones(C_y, dtype=torch.float32)
            log.warning(
                "No `residual_norm` in metadata; assuming the residual was not "
                "standardized (mean=0, std=1)."
            )
        self._res_mean = mean.view(1, C_y, 1, 1).to(self.device)
        self._res_std = std.view(1, C_y, 1, 1).to(self.device)

        log.info("CPMGEM-residual downscaler ready")

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
        Override base ``downscale``: regressor mean + reverse sub-VP residual sampling.

        Loop structure mirrors downscaler_cpmgem / downscaler_resdiff
        (date-outer, member-inner):
          - Preprocess c_low ONCE per date batch.
          - Run the regressor ONCE → ŷ_det (cond_high, shared across members).
          - Per member: sample a standardized residual via the reverse sub-VP chain,
            un-standardize, add ŷ_det, denormalize, postprocess.
        """
        if verbose:
            log.info("Starting CPMGEM-residual downscaling process")

        if model is None:
            model = self.model
        self._amp_dtype = self._parse_amp_dtype(amp_dtype)
        model = self._maybe_compile(model, compile)
        model.eval()

        all_dates_np = [np.datetime64(d) for d in self.target_dates]
        T = len(self.target_dates)
        M = self.ensemble_size
        n_batches = (T + batch_size - 1) // batch_size

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

            # ── Preprocess + normalize low-res conditioning once per date batch ─
            c_low = self._stack_to_device(
                [self._preprocess_single_date(d) for d in batch_dates]
            )  # (B, C_x, H_x, W_x)
            if self.norm_x is not None:
                c_low = self.norm_x(c_low)

            # ── Deterministic regressor mean (shared across members) ───────────
            # The regressor is a SongUNet used deterministically: x and t are zeros,
            # the conditioning enters via cond_low (matches downscaler_resdiff and
            # pydataset_resdiff's residuals build).
            with torch.inference_mode(), self._amp_ctx():
                B_cur = c_low.shape[0]
                x_dummy = torch.zeros(
                    B_cur, len(self.vars_y), self.H_y, self.W_y, device=self.device
                )
                t_dummy = torch.zeros(B_cur, device=self.device)
                c_high = self.regressor(
                    x=x_dummy, t=t_dummy, cond_low=c_low
                )  # (B, C_y, H_y, W_y) — ŷ_det in normalized space

            # ── High-res forcings (cond_high = [ŷ_det, f]) ─────────────────────
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

            # ── Stochastic residual: one reverse sub-VP chain per member ───────
            for member in range(M):
                with self._amp_ctx():
                    # Inherited reverse sub-VP sampler; f_cond routes ŷ_det into
                    # the network's cond_high stream. Sample is the *standardized*
                    # residual (unit-variance space the model was trained in).
                    r_std = self.sample(
                        x_cond=c_low, model=model, f_cond=c_high
                    )  # (B, C_y, H_y, W_y)
                # Un-standardize, then add the regressor mean → normalized y-space.
                r_hat = r_std * self._res_std + self._res_mean
                p_gpu = c_high + r_hat
                if self.norm_y is not None:
                    p_gpu = self.norm_y.inverse_transform(p_gpu.float())
                p_cpu = self._async_d2h(p_gpu.float())
                if self._cuda:
                    torch.cuda.synchronize()
                member_buffers[member].append(self._postprocess_numpy(p_cpu.numpy()))
                del r_std, r_hat, p_gpu, p_cpu

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
