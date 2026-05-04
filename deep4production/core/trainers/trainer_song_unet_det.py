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
from deep4production.core.trainers.trainer import trainer
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

    def __init__(self, data, dataloader, id_dir, model_info, graph, d4dpy, Mlflow,
                 normalizer_info_x=None, normalizer_info_y=None, normalizer_info_f=None):
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
        )
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
                        f : forcing or "N/A" sentinel (unused)
        optimizer     : torch.optim.Optimizer
        loss_function : any d4p loss with forward(target, output)
        device        : str
        is_this_training : bool
        """
        x, y, _ = data
        non_blocking = (self.device == "cuda")
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        B = y.shape[0]

        # --- GPU-side normalization ---
        x, y, _ = self._normalize_inputs(x=x, y=y)

        t    = torch.zeros(B, device=device)
        x_in = torch.zeros_like(y)  # model conditions entirely via cond_low

        optimizer.zero_grad(set_to_none=True)

        # Forward + loss under AMP autocast when enabled (bf16/fp16).
        with self._amp_ctx():
            prediction = model(x=x_in, t=t, cond_low=x)
            loss = loss_function(target=y, output=prediction)

        if is_this_training:
            if self._scaler is not None:
                self._scaler.scale(loss).backward()
            else:
                loss.backward()

        return loss.detach()
