"""
GPU-resident batch normalizer (anemoi-style).

Replaces the per-sample CPU normalization loop in ``pydataset.preprocess``
with a single fused vectorized in-place op on the GPU tensor. Per-variable
strategies are baked into two flat ``(C,)`` buffers (``_norm_mul`` and
``_norm_add``) at init, so ``forward()`` is one ``mul_().add_()`` call
regardless of how many strategies are mixed across channels.

All five d4p normalization strategies (``mean_std``, ``std``, ``max``,
``minmax_neg1_1``, ``none``) are affine maps and collapse onto this
representation exactly.

Operators (e.g. ``sqrt`` for precipitation) remain on the CPU side because
they are non-linear. Their effect on the affine coefficients is propagated
via the ``stats_transform`` mechanism that already exists in
``d4pnormalizers.minmax_neg1_1``.

Authors:
    Jorge Baño-Medina
"""

import importlib
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn

from deep4production.utils.log import get_logger

log = get_logger("deep.preprocessing.normalizer")

_SUPPORTED_METHODS = ("none", "mean_std", "std", "max", "minmax_neg1_1")


class InputNormalizer(nn.Module):
    """
    Vectorized affine normalizer applied as the model's first preprocessor.

    Maps every per-variable d4p normalization strategy onto the affine form
    ``y = mul * x + add`` and stores the per-channel coefficients as two
    registered buffers. The forward pass is a single fused in-place op on
    the GPU tensor.

    Parameters
    ----------
    normalizer_info : dict
        The dict produced by ``pydataset._resolve_normalizer_info()`` for
        predictors, predictands or forcings (also persisted under
        ``metadata["normalizer_x"|"_y"|"_f"]`` in the checkpoint). Must contain:

        - ``"kwargs"``: ``{var_name: {"mean", "std", "min", "max",
          "stats_transform" (optional)}}``
        - ``"normalizer_func_per_variable"``: ``{var_name: method_name}``
          where ``method_name`` is one of the d4p strategy names listed in
          ``_SUPPORTED_METHODS``.
    vars : Iterable[str]
        Variable names in channel order (matches the tensor's channel axis).
    channel_dim : int, default 1
        Which axis is the channel axis. d4p's convention is ``(B, C, ...)``
        so the default is 1. Use ``2`` for lagged tensors of shape
        ``(B, L, C, ...)``.

    Notes
    -----
    Behaviour-preserving relative to the legacy ``d4pnormalizers``: only
    ``minmax_neg1_1`` consumes the ``stats_transform`` hint. The other
    methods read raw zarr stats unchanged. If you combine an operator
    with ``mean_std`` / ``std``, a warning is logged because the raw
    mean/std are no longer correct after the operator (this matches the
    silent legacy behaviour but makes it visible).

    Examples
    --------
    The trainer (deep4production.core.trainers.trainer) builds and applies
    these automatically — see ``trainer._normalize_inputs`` for the wiring.
    Below is the equivalent manual usage if you write a custom trainer::

        from deep4production.core.pydatasets.pydataset import pydataset
        from deep4production.deep.preprocessing.normalizer import InputNormalizer

        resolved = pydataset._resolve_normalizer_info(
            normalizer_info_recipe, vars_x, predictand=False
        )
        norm_x = InputNormalizer(resolved, vars_x).to(device)

        for x, y, f in dataloader:
            x = x.to(device, non_blocking=True)
            x = norm_x(x)                                # fused mul_/add_
            y_hat = model(x, ...)
            y_hat = norm_y.inverse_transform(y_hat)      # for denormed preds
    """

    def __init__(
        self,
        normalizer_info: dict,
        vars: Iterable[str],
        channel_dim: int = 1,
    ) -> None:
        super().__init__()
        if normalizer_info is None:
            raise ValueError(
                "normalizer_info is None — nothing to do. "
                "Skip InputNormalizer entirely for this block."
            )
        if channel_dim < 0:
            raise ValueError("channel_dim must be a non-negative axis index.")

        method_per_var = normalizer_info["normalizer_func_per_variable"]
        kwargs_per_var = normalizer_info["kwargs"]

        vars = list(vars)
        C = len(vars)
        mul = np.ones(C, dtype=np.float32)
        add = np.zeros(C, dtype=np.float32)

        for i, var in enumerate(vars):
            method = method_per_var.get(var)
            stats = kwargs_per_var[var]

            mean_i = float(stats["mean"])
            std_i = float(stats["std"])
            min_i = float(stats["min"])
            max_i = float(stats["max"])
            stats_transform = stats.get("stats_transform")

            if method is None or method == "none":
                continue

            if method == "mean_std":
                if stats_transform is not None:
                    log.warning(
                        "Variable '%s' uses 'mean_std' with operator '%s'. "
                        "The raw zarr mean/std are not corrected for the operator; "
                        "use 'minmax_neg1_1' or recompute stats on operator-applied data.",
                        var, stats_transform,
                    )
                if std_i == 0.0:
                    raise ValueError(f"std=0 for variable '{var}'; cannot mean_std-normalize.")
                mul[i] = 1.0 / std_i
                add[i] = -mean_i / std_i

            elif method == "std":
                if stats_transform is not None:
                    log.warning(
                        "Variable '%s' uses 'std' with operator '%s'. "
                        "The raw zarr std is not corrected for the operator.",
                        var, stats_transform,
                    )
                if std_i == 0.0:
                    raise ValueError(f"std=0 for variable '{var}'; cannot std-normalize.")
                mul[i] = 1.0 / std_i
                add[i] = 0.0

            elif method == "max":
                if max_i == 0.0:
                    raise ValueError(f"max=0 for variable '{var}'; cannot max-normalize.")
                mul[i] = 1.0 / max_i
                add[i] = 0.0

            elif method == "minmax_neg1_1":
                if stats_transform is not None:
                    ops = importlib.import_module("deep4production.utils.operators")
                    fn = getattr(ops, stats_transform)
                    min_i = float(fn(min_i))
                    max_i = float(fn(max_i))
                rng = max_i - min_i
                if rng == 0.0:
                    raise ValueError(
                        f"max == min for variable '{var}'; cannot minmax_neg1_1-normalize."
                    )
                # y = 2*(x - min)/rng - 1 = (2/rng)*x - (2*min/rng + 1)
                mul[i] = 2.0 / rng
                add[i] = -2.0 * min_i / rng - 1.0

            else:
                raise ValueError(
                    f"Unknown normalization method '{method}' for variable '{var}'. "
                    f"Supported: {_SUPPORTED_METHODS}"
                )

        # Persistent so the affine coefficients are saved in the checkpoint
        # alongside the model weights — guarantees inference normalization
        # matches training normalization regardless of recipe drift.
        self.register_buffer("_norm_mul", torch.from_numpy(mul), persistent=True)
        self.register_buffer("_norm_add", torch.from_numpy(add), persistent=True)

        self._vars = vars
        self._channel_dim = channel_dim

    @property
    def vars(self) -> list:
        return list(self._vars)

    @property
    def channel_dim(self) -> int:
        return self._channel_dim

    def _resolve_channel_dim(self, channel_dim) -> int:
        return self._channel_dim if channel_dim is None else int(channel_dim)

    def _broadcast_shape(self, ndim: int, channel_dim: int) -> tuple:
        if channel_dim >= ndim:
            raise ValueError(
                f"channel_dim={channel_dim} is out of range for tensor of "
                f"ndim={ndim}."
            )
        shape = [1] * ndim
        shape[channel_dim] = -1
        return tuple(shape)

    def _check_channels(self, x: torch.Tensor, channel_dim: int) -> None:
        C_expected = self._norm_mul.numel()
        C_got = x.shape[channel_dim]
        if C_got != C_expected:
            raise ValueError(
                f"InputNormalizer expected C={C_expected} channels at dim "
                f"{channel_dim} but got x.shape={tuple(x.shape)}."
            )

    def transform(
        self,
        x: torch.Tensor,
        in_place: bool = True,
        channel_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply the affine normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with channel axis at ``channel_dim`` (or
            ``self.channel_dim`` if not given).
        in_place : bool, default True
            Modify ``x`` in place to save memory. Set False when the caller
            still needs the un-normalized tensor afterwards.
        channel_dim : int, optional
            Per-call override of the channel axis. Useful for trainers that
            see the same variables in two different layouts (e.g. GNN4CD
            uses ``(B, C, G)`` for non-lagged and ``(B, L, C, G)`` for lagged
            inputs, with channel at dim 1 vs 2).

        Returns
        -------
        torch.Tensor
            Normalized tensor (same object as ``x`` if ``in_place=True``).
        """
        cd = self._resolve_channel_dim(channel_dim)
        self._check_channels(x, cd)
        if not in_place:
            x = x.clone()
        shape = self._broadcast_shape(x.ndim, cd)
        x.mul_(self._norm_mul.view(shape)).add_(self._norm_add.view(shape))
        return x

    def inverse_transform(
        self,
        x: torch.Tensor,
        in_place: bool = True,
        channel_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Invert the affine normalization. Use this on model predictions
        before metrics or before writing to disk.
        """
        cd = self._resolve_channel_dim(channel_dim)
        self._check_channels(x, cd)
        if not in_place:
            x = x.clone()
        shape = self._broadcast_shape(x.ndim, cd)
        x.sub_(self._norm_add.view(shape)).div_(self._norm_mul.view(shape))
        return x

    def forward(self, x: torch.Tensor, channel_dim: Optional[int] = None) -> torch.Tensor:
        return self.transform(x, in_place=True, channel_dim=channel_dim)
