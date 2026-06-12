"""
Downscaler for EDM-direct — inference for the EDM × direct cell.

Single-stage: an EDM-preconditioned SongUNet generates the full normalized field
directly with the EDM Heun sampler (Karras et al. 2022) — no regressor, no
residual. It is the EDM analogue of the CPMGEM-direct downscaler (which uses the
reverse sub-VP chain instead).

    y = EDM_Heun(cond_low=x, cond_high=f)        (in normalized / operator space)

The Heun sampler itself (`_sigmas`, `_edm_heun_sample`, `sample`) is inherited
unchanged from `downscaler_resdiff`; for the direct cell its output is the full
field rather than a residual, so there is nothing to add afterwards. Conditioning
is cond_low (predictors) + optional cond_high (forcings, e.g. orography), matching
the EDM-direct trainer.

Extra constructor argument (via ``d4p_downscaler.kwargs`` in the inference YAML):

    sampling_params (dict, optional):
        num_steps / sigma_min / sigma_max / rho / S_churn / S_min / S_max / S_noise
        — exactly as documented in downscaler_resdiff.

Author:
    Jorge Baño-Medina
"""

import numpy as np
import torch
import xarray as xr

from deep4production.core.downscalers.downscaler import downscaler
from deep4production.core.downscalers.downscaler_resdiff import (
    downscaler_custom as downscaler_resdiff,
)
from deep4production.utils.trans import from_pred_to_xarray
from deep4production.utils.log import get_logger

log = get_logger("downscaler.edm_direct")


class downscaler_custom(downscaler_resdiff):
    """
    EDM-direct downscaler.

    Reuses ResDiff's EDM Heun sampler but has no regressor and samples the full
    field directly. ``__init__`` deliberately calls the *base* downscaler
    initializer (not ResDiff's, which loads a mandatory regressor) and then parses
    the EDM sampler parameters; ``downscale`` is overridden for single-stage
    direct generation.
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
        # Skip downscaler_resdiff.__init__ (it requires a regressor) and call the
        # base initializer directly; we only want ResDiff's *sampler* methods.
        downscaler.__init__(
            self,
            id_dir=id_dir,
            input_data=input_data,
            model_file=model_file,
            saving_info=saving_info,
            ensemble_size=ensemble_size,
            graph=graph,
            forcing_data=forcing_data,
        )

        # ── EDM Heun sampler parameters (mirrors downscaler_resdiff.__init__) ──
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

        log.info(
            "EDM-direct sampler: steps=%d sigma=[%g, %g] rho=%g S_churn=%g",
            self.num_steps,
            self.sigma_min,
            self.sigma_max,
            self.rho,
            self.S_churn,
        )

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
        Override base ``downscale``: single-stage EDM Heun sampling of the full
        field. Date-outer, member-inner:
          - Preprocess + normalize c_low (and optional cond_high forcings) ONCE
            per date batch.
          - Per member: draw a fresh full-field sample, denormalize, postprocess.
        """
        if verbose:
            log.info("Starting EDM-direct downscaling process")

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

            # ── Optional high-res forcings (cond_high), e.g. orography ──────────
            f_cond = None
            if self.forcing_data is not None:
                f_cond = self._stack_to_device(
                    [self._preprocess_forcing_date(d) for d in batch_dates]
                )  # (B, C_f, H_y, W_y)
                if self.norm_f is not None:
                    f_cond = self.norm_f(f_cond)

            # ── EDM Heun sampling: one full-field draw per member ──────────────
            for member in range(M):
                with self._amp_ctx():
                    # Inherited ResDiff sampler; for the direct cell the output is
                    # the full normalized field, not a residual.
                    p_torch = self.sample(
                        c_low=c_low, c_high=f_cond, model=model
                    )  # (B, C_y, H_y, W_y)
                if self.norm_y is not None:
                    p_torch = self.norm_y.inverse_transform(p_torch.float())
                p_cpu = self._async_d2h(p_torch.float())
                if self._cuda:
                    torch.cuda.synchronize()
                member_buffers[member].append(self._postprocess_numpy(p_cpu.numpy()))
                del p_torch, p_cpu

            del c_low, f_cond

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
