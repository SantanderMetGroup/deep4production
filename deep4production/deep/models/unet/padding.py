"""
Reflection padding for whole-domain training on grids that are not divisible by
2^(len(ch_mult)-1).

A SongUNet halves its spatial dims once per level with `avg_pool2d` (floor) and
doubles them back with `interpolate` (exact), so a decoder skip-concat only lines
up when H and W are multiples of 2^(num_levels-1). CORDEX grids rarely are (e.g.
EUR-12i 341x352: 341 = 11*31 is odd and cannot be halved even once).

The padder reflection-pads the predictand grid up to the next multiple, runs the
model there, and crops the prediction back BEFORE the loss — so the padded rim
never contributes a gradient and no loss function needs a mask. Predictors are
upsampled to the NATIVE grid first and padded afterwards, which keeps every
predictor pixel over its true location (interpolating straight to the padded
extent would stretch the low-res field over a domain it does not cover).

Not needed with `patching`: `GridPatching2D` already reflection-pads internally
to build a full patch cover, so the two are mutually exclusive.

Author:
    Jorge Baño-Medina
"""

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────
def pad_plan(H: int, W: int, multiple: int):
    """(left, right, top, bottom) reflection pad centring the fill on each axis."""
    m = int(multiple)
    if m < 1:
        raise ValueError(f"pad_to_multiple must be >= 1, got {m}.")
    ph, pw = (-int(H)) % m, (-int(W)) % m
    return (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)


# ──────────────────────────────────────────────────────────────────────────
class ReflectPadder:
    """
    Reflection padder for a fixed (H, W) predictand grid.

    `apply`/`apply_cond_low` move tensors onto the padded grid the model was
    trained on; `crop` brings a model output back to the native grid.
    `crop(apply(x)) == x` exactly.
    """

    def __init__(self, img_shape, multiple: int):
        self.H, self.W = int(img_shape[0]), int(img_shape[1])
        self.multiple = int(multiple)
        self.amounts = pad_plan(self.H, self.W, self.multiple)
        l, r, t, b = self.amounts
        # F.pad(mode="reflect") requires each pad to be strictly smaller than the
        # corresponding dimension.
        if max(l, r) >= self.W or max(t, b) >= self.H:
            raise ValueError(
                f"pad_to_multiple={self.multiple} needs pad {self.amounts} on a "
                f"{self.H}x{self.W} grid, which exceeds what reflection padding "
                f"allows. Use a smaller multiple (a shallower ch_mult) or crop."
            )
        self.padded_H, self.padded_W = self.H + t + b, self.W + l + r
        self.active = any(self.amounts)

    # -- geometry ---------------------------------------------------------
    def padded_shape(self, *leading):
        """(*leading, padded_H, padded_W) — for allocating the model's x slot."""
        return (*leading, self.padded_H, self.padded_W)

    def as_config(self) -> dict:
        """Serializable geometry persisted in run metadata for inference."""
        return {
            "enabled": True,
            "pad_to_multiple": self.multiple,
            "amounts": list(self.amounts),
            "H": self.H,
            "W": self.W,
            "padded_H": self.padded_H,
            "padded_W": self.padded_W,
        }

    # -- tensor ops -------------------------------------------------------
    def apply(self, field):
        """Reflection-pad a predictand-grid tensor (..., H, W). None passes through."""
        if field is None or not self.active:
            return field
        if tuple(field.shape[-2:]) != (self.H, self.W):
            raise ValueError(
                f"padder expects a {self.H}x{self.W} field, got "
                f"{tuple(field.shape[-2:])}."
            )
        return F.pad(field, self.amounts, mode="reflect")

    def apply_cond_low(self, x):
        """
        Bring native low-res predictors onto the padded grid: bilinear upsample to
        (H, W) FIRST, then reflection-pad, so the predictors stay geographically
        aligned with the target (see module docstring).
        """
        if x is None or not self.active:
            return x
        if x.dim() != 4:
            raise ValueError(
                f"padded cond_low must be (B, C, H, W); got {tuple(x.shape)}. "
                "Lagged predictors must be flattened onto the channel dim first."
            )
        x = F.interpolate(
            x, size=(self.H, self.W), mode="bilinear", align_corners=False
        )
        return F.pad(x, self.amounts, mode="reflect")

    def crop(self, field):
        """Crop a padded-grid tensor (..., padded_H, padded_W) back to (H, W)."""
        if field is None or not self.active:
            return field
        if tuple(field.shape[-2:]) != (self.padded_H, self.padded_W):
            raise ValueError(
                f"padder expects a {self.padded_H}x{self.padded_W} field to crop, "
                f"got {tuple(field.shape[-2:])}."
            )
        l, _, t, _ = self.amounts
        return field[..., t : t + self.H, l : l + self.W]


# ──────────────────────────────────────────────────────────────────────────
def build_padder(cfg, img_shape):
    """
    Build a `ReflectPadder` from either a recipe value (`pad_to_multiple: 16`) or
    a persisted `padding` metadata block. Returns None when padding is disabled
    or when the grid already divides evenly (so callers keep their fast path).
    """
    if cfg is None or cfg is False:
        return None
    if isinstance(cfg, dict):
        if not cfg.get("enabled", False):
            return None
        multiple = cfg["pad_to_multiple"]
        # A persisted block records the training grid: refuse to replay it on a
        # different one (the recomputed pad would silently differ from training).
        if "H" in cfg and (int(cfg["H"]), int(cfg["W"])) != tuple(map(int, img_shape)):
            raise ValueError(
                f"padding metadata was built for a {cfg['H']}x{cfg['W']} grid but "
                f"the current predictand grid is {img_shape[0]}x{img_shape[1]}."
            )
    else:
        multiple = cfg
    if int(multiple) <= 1:
        return None
    padder = ReflectPadder(img_shape, multiple)
    return padder if padder.active else None
