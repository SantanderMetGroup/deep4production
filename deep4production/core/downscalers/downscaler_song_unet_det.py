"""
Deterministic SongUNet downscaler for precipitation downscaling.

Mirrors the inference-side logic of trainer_song_unet_det:
  - x_in = zeros(1, C_y, H_y, W_y)  — no noise
  - t    = zeros(1,)                  — noise label fixed at 0
  - cond_low = preprocessed low-resolution predictor fields

All preprocessing (operators, normalizers, 2D reshape) and postprocessing
(denormalise, de-operator, xarray formatting) are inherited from the base
downscaler unchanged.

Authors:
    Jorge Baño-Medina
"""

import numpy as np
import torch
from deep4production.core.downscalers.downscaler import downscaler
from deep4production.utils.trans import from_pred_to_xarray
from deep4production.utils.log import get_logger

log = get_logger("downscaler.songunet")


class downscaler_custom(downscaler):
    """
    Deterministic SongUNet downscaler.

    Inherits all dataset / preprocessing / postprocessing logic from the base
    downscaler.  Only ``downscale`` is overridden to call the SongUNet with
    zeros for the noisy-input slot and a constant t = 0.
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
        log.info("Deterministic SongUNet downscaler ready")

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
        Deterministic SongUNet inference loop. Date-outer; no member loop
        (model is deterministic, ensemble_size > 1 is broadcast at the end).

        Parameters
        ----------
        batch_size : int
        amp_dtype  : 'bfloat16' / 'float16' / None
        compile    : bool — wrap model with torch.compile(dynamic=True)
        """
        if verbose:
            log.info("Starting deterministic SongUNet downscaling process")

        if model is None:
            model = self.model
        self._amp_dtype = self._parse_amp_dtype(amp_dtype)
        model = self._maybe_compile(model, compile)
        model.eval()

        C_y = len(self.vars_y)
        spatial = [self.H_y, self.W_y] if self.transform_to_2D_y else [self.G_y]

        all_dates_np = [np.datetime64(d) for d in self.target_dates]
        T = len(self.target_dates)
        n_batches = (T + batch_size - 1) // batch_size

        # -- Pipelined date loop (async D2H overlap) --
        all_preds = []
        pending_cpu = None
        for b_idx in range(n_batches):
            i = b_idx * batch_size
            batch_dates = self.target_dates[i : i + batch_size]
            B = len(batch_dates)
            if verbose:
                log.info(
                    "Batch %d/%d: %s → %s (%d dates)",
                    b_idx + 1,
                    n_batches,
                    batch_dates[0],
                    batch_dates[-1],
                    B,
                )

            # ── Preprocess low-res conditioning ──────────────────────────
            inp = self._stack_to_device(
                [self._preprocess_single_date(d) for d in batch_dates]
            )  # (B, C_x, H_x, W_x)

            # ── GPU-side input normalization (mirrors trainer_song_unet_det) ──
            # Deterministic models always condition on normalized predictors;
            # the trainer applies _normalize_inputs(x=x, y=y) and inference must
            # match. norm_x is None only if the user trained without an
            # x-normalizer, which is non-standard for these architectures.
            if self.norm_x is not None:
                inp = self.norm_x(inp)

            # ── Deterministic forward pass ────────────────────────────────
            t = torch.zeros(B, device=self.device)
            x_in = torch.zeros(B, C_y, *spatial, device=self.device)
            with torch.inference_mode(), self._amp_ctx():
                p_torch = model(x=x_in, t=t, cond_low=inp)

            # ── GPU-side denormalization of the prediction ─────────────────
            # Predictand normalization is loss-dependent: MseLoss recipes
            # normalize y (norm_y is set), Asym/NLL recipes operate in raw
            # operator-space (norm_y is None). The guard preserves both paths.
            if self.norm_y is not None:
                p_torch = self.norm_y.inverse_transform(p_torch.float())

            # ── Async D2H + flush previous ────────────────────────────────
            if pending_cpu is not None:
                if self._cuda:
                    torch.cuda.synchronize()
                all_preds.append(self._postprocess_numpy(pending_cpu.numpy()))
            pending_cpu = self._async_d2h(p_torch.float())
            del inp, x_in, p_torch

        # Final flush
        if pending_cpu is not None:
            if self._cuda:
                torch.cuda.synchronize()
            all_preds.append(self._postprocess_numpy(pending_cpu.numpy()))

        # ── Build xarray ONCE; broadcast across ensemble dim ─────────────
        all_preds_np = np.concatenate(all_preds, axis=0)  # (T, C, G)
        ds = from_pred_to_xarray(
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
        if self.ensemble_size > 1:
            ds = ds.expand_dims(member=np.arange(self.ensemble_size))
        else:
            ds = ds.expand_dims(member=[0])

        if self.format_output:
            ds = self.formatting_func(ds, **self.formatting_kwargs)
        if return_pred:
            return ds
        log.debug("Writing prediction xarray to %s\n%s", self.output_path, ds)
        ds.to_netcdf(self.output_path)
