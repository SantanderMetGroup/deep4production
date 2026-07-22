"""
Deterministic SongUNet trainer for precipitation downscaling.

The SongUNet backbone is repurposed as a pure regression model by bypassing
the diffusion process entirely:
  - x_in (the "noisy input" slot) is filled with zeros — no noise is ever added
  - t is set to zeros — the sinusoidal noise embedding collapses to a fixed vector
  - cond_low carries the normalised low-resolution predictor fields

This preserves the full UNet capacity (multi-scale ResBlocks, FIR resampling,
self-attention) while making training deterministic and compatible with any
standard regression loss (Asym, MseLoss, MaeLoss, ...).

Authors:
    Jorge Baño-Medina
"""

import torch
import torch.nn.functional as F
from deep4production.core.trainers.trainer import trainer
from deep4production.deep.models.diffusion.patching import (
    build_train_patcher,
    assemble_cond_patches,
    validate_patched_cond_high,
)
from deep4production.utils.log import get_logger

log = get_logger("trainer.songunet")


class trainer_custom(trainer):
    """
    Deterministic SongUNet trainer.

    Inherits all dataset / dataloader / MLflow logic from the base trainer.
    Only ``model_backprop`` is overridden to wire up the SongUNet signature
    (x, t, cond_low) instead of the default (x, f) used by CNN-based models.

    No extra YAML keys are required beyond the standard model_info block.
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

        # --- Optional CorrDiff-style patching (large-domain tiling) -----------
        # Enabled via model_info.training_params.patching.enabled. When on, the
        # deterministic regressor is trained on random P×P patches with a GLOBAL
        # positional embedding and cond_high = [img_lr_hr | forcing_hr | global_lr].
        # The geometry is persisted to metadata so the downscaler (and the
        # resdiff pydataset/downscaler that reuse this regressor) tile identically.
        self.patcher = None
        self.K_pe = 0
        self._patch_validated = False
        patch_cfg = (model_info.get("training_params") or {}).get("patching")
        if patch_cfg and patch_cfg.get("enabled", False):
            self.patcher, self.K_pe, resolved_cfg = build_train_patcher(
                patch_cfg, model_info["model_params"]["kwargs"]
            )
            self.metadata_dict["patching"] = resolved_cfg
            log.info("Patched regressor training enabled: %s", resolved_cfg)

        log.info("Deterministic SongUNet trainer ready")

    # ─────────────────────────────────────────────────────────────────────────
    def model_backprop(
        self,
        model,
        data,
        optimizer,
        loss_function,
        device,
        is_this_training=True,
        **kwargs,
    ):
        """
        One forward / backward pass for deterministic SongUNet.

        SongUNet forward signature: (x, t, cond_low, cond_high).
        Deterministic mapping:
            x_in    = zeros(B, C_y, H_y, W_y)   — no noisy sample to denoise
            t       = zeros(B,)                   — noise label fixed at 0
            cond_low = x predictor                — low-res conditioning

        Parameters
        ----------
        model         : SongUNet instance
        data          : (x, y, f) from standard pydataset DataLoader
                        x : (B, C_x, H_x, W_x)  low-res predictor fields
                        y : (B, C_y, H_y, W_y)  high-res precipitation target
                        f : (B, C_f, H_y, W_y) high-res forcing (cond_high, e.g.
                            orography) or "N/A" sentinel when no forcings are
                            configured
        optimizer     : torch.optim.Optimizer
        loss_function : any d4p loss with forward(target, output)
        device        : str
        is_this_training : bool
        """
        x, y, f = data
        non_blocking = self.device_type == "cuda"
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        B = y.shape[0]

        # High-res conditioning (cond_high), e.g. static orography. pydataset
        # emits the "N/A" string sentinel when no forcings are configured; only a
        # real tensor is routed to the model. This keeps recipes without forcings
        # (cond_high_channels=0) working unchanged, where cond_high stays None.
        # NOTE: test for a tensor rather than for the sentinel -- the DataLoader's
        # default collate turns the per-sample "N/A" strings into a LIST of B
        # strings, so ``isinstance(f, str)`` is False for a batch and the code
        # would fall through to ``f.to(device)`` on a list.
        use_cond_high = torch.is_tensor(f)
        if use_cond_high:
            f = f.to(device, non_blocking=non_blocking)

        # --- GPU-side normalization (predictors + predictands + forcings) ---
        # The forcing f (e.g. orography) is normalized with its own
        # InputNormalizer (norm_f) when a forcing normalizer is configured.
        if use_cond_high:
            x, y, f = self._normalize_inputs(x=x, y=y, f=f)
        else:
            x, y, _ = self._normalize_inputs(x=x, y=y)

        optimizer.zero_grad(set_to_none=True)

        # ── Patched (CorrDiff-style) branch ──────────────────────────────────
        if self.patcher is not None:
            loss = self._patched_forward_loss(
                model, x, y, f if use_cond_high else None, loss_function
            )
        else:
            # ── Whole-domain branch (unchanged) ──────────────────────────────
            t = torch.zeros(B, device=device)
            x_in = torch.zeros_like(y)  # model conditions entirely via cond_low/high
            # Forward + loss under AMP autocast when enabled (bf16/fp16). cond_high
            # carries any high-res forcing already at predictand resolution (None
            # when no forcings are configured); cond_low is the low-res stream.
            with self._amp_ctx():
                prediction = model(
                    x=x_in, t=t, cond_low=x, cond_high=f if use_cond_high else None
                )
                loss = loss_function(target=y, output=prediction)

        if is_this_training:
            if self._scaler is not None:
                self._scaler.scale(loss).backward()
            else:
                loss.backward()

        return loss.detach()

    # ─────────────────────────────────────────────────────────────────────────
    def _patched_forward_loss(self, model, x, y, f, loss_function):
        """
        Patched deterministic forward + loss. Predictors `x` (native LR) are
        bilinearly upsampled to the HR grid, then random P×P patches are drawn;
        each patch is conditioned on cond_high = [img_lr_hr | forcing_hr | global_lr]
        plus a global positional embedding, and the loss is computed directly on
        the patched target (spatially-invariant losses only — the per-gridpoint
        Asym loss is unsupported here).
        """
        B, C_y, H, W = y.shape
        C_x = x.shape[1]
        C_f = f.shape[1] if f is not None else 0

        if not self._patch_validated:
            validate_patched_cond_high(
                self.model_params["kwargs"]["cond_high_channels"],
                C_x, C_y, C_f, stage="regressor",
            )
            if torch.is_tensor(getattr(loss_function, "shape", None)):
                raise ValueError(
                    "Patched regressor training needs a spatially-invariant loss "
                    "(e.g. MseLoss / MaeLoss). The configured loss carries "
                    "per-gridpoint parameters (Asym-style) that cannot be cropped "
                    "to random patches."
                )
            self._patch_validated = True

        x_hr = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        cond_local = x_hr if f is None else torch.cat([x_hr, f], dim=1)

        self.patcher.new_origins(H, W, y.device)
        cond_high, pos_embd = assemble_cond_patches(self.patcher, cond_local, x_hr, self.K_pe)
        y_patches = self.patcher.extract(y)
        Pn = self.patcher.patch_num
        x_in = torch.zeros(Pn * B, C_y, self.patcher.Py, self.patcher.Px,
                           device=y.device, dtype=y.dtype)
        t = torch.zeros(Pn * B, device=y.device)

        with self._amp_ctx():
            prediction = model(
                x=x_in, t=t, cond_low=None, cond_high=cond_high, pos_embd=pos_embd
            )
            return loss_function(target=y_patches, output=prediction)
