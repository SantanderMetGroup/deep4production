"""
Patch-based (multi-)diffusion utilities for CorrDiff-style downscaling on large
domains (Mardani et al. 2023; MultiDiffusion, Bar-Tal et al. 2023).

This is a self-contained port of the functional Modulus / PhysicsNeMo
`image_batching` / `image_fuse` algorithm (the framework itself is not a
dependency here). It provides two patchers and a couple of helpers shared by the
RESDIFF trainers / downscalers and the deterministic-SongUNet regressor stage:

  - `GridPatching2D`   : deterministic full-cover patch grid with an
                         accumulate-and-average `fuse` (used at inference and for
                         the deterministic regressor tiling).
  - `RandomPatching2D` : `patch_num` random patches per step (used at training,
                         mirroring CorrDiff's patched training loop). No fuse.
  - `build_spatial_pe` : the NeRF-style spatial positional embedding, factored
                         out of `SongUNet` so the GLOBAL positional embedding a
                         patcher gathers is byte-identical to what the model
                         would compute over the full grid.
  - `assemble_cond_patches` / `run_regressor_patched` : assemble the per-patch
                         conditioning `[local crops | global_lr thumbnail]` plus
                         the per-patch global positional embedding.

Conditioning layout for a patched SongUNet (see corrdiff spec §2). All
conditioning is delivered at the HR patch grid through `cond_high`
(`cond_low_channels == 0` for a patched model). Per patch (P×P):

    diffusion : x = residual crop (C_y);
                cond_high = [mean_hr (C_y) | img_lr_hr (C_x) | global_lr (C_x)]
    regressor : x = zeros (C_y);
                cond_high = [img_lr_hr (C_x) | forcing_hr (C_f) | global_lr (C_x)]
    both      : pos_embd = global spatial PE gathered per patch (4·K)

`img_lr_hr` is the native low-res predictor stack bilinearly upsampled to the HR
grid ONCE before patching; `global_lr` is that whole field bilinearly resized to
P×P and appended on the channel dim of every patch.

Author:
    Jorge Baño-Medina
"""

import math

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────
# Spatial positional embedding (shared with SongUNet)
# ──────────────────────────────────────────────────────────────────────────
def build_spatial_pe(H: int, W: int, K: int, device, dtype) -> torch.Tensor:
    """
    Build a (1, 4·K, H, W) NeRF-style spatial positional encoding.

    For k ∈ [0, K-1] the channels are, in order,
        sin(2^k · 2π · y/H), cos(2^k · 2π · y/H),
        sin(2^k · 2π · x/W), cos(2^k · 2π · x/W).
    K = 1 matches the 4-channel spatial PE used in CorrDiff (Mardani et al. 2023).

    This is the single source of truth for the PE; `SongUNet._build_spatial_pe`
    delegates here, and the patchers build the full-domain PE with this same
    function and then gather per-patch crops — so a patch's PE is exactly the
    slice of the full-domain PE at its location (the "global_index gather").
    """
    y = torch.arange(H, device=device, dtype=dtype).view(H, 1) / H  # (H, 1)
    x = torch.arange(W, device=device, dtype=dtype).view(1, W) / W  # (1, W)
    freqs = (2.0 ** torch.arange(K, device=device, dtype=dtype)) * (2.0 * math.pi)
    chans = []
    for f in freqs:
        chans.append((f * y).sin().expand(H, W))
        chans.append((f * y).cos().expand(H, W))
        chans.append((f * x).sin().expand(H, W))
        chans.append((f * x).cos().expand(H, W))
    return torch.stack(chans, dim=0).unsqueeze(0)  # (1, 4K, H, W)


# ──────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────
def validate_patch_shape(patch_shape, num_levels: int) -> None:
    """
    A SongUNet with `num_levels` levels halves resolution `num_levels-1` times, so
    each spatial dim it processes must be a multiple of 2^(num_levels-1); otherwise
    encoder-floored sizes don't line up with decoder-doubled sizes at the skip
    connections (shape-mismatch crash). The patch shape is what the net actually
    sees, so it must satisfy the constraint (patching does NOT fix divisibility of
    the full grid — that must be handled by resizing the data grid).
    """
    div = 2 ** (num_levels - 1)
    py, px = patch_shape
    if py % div != 0 or px % div != 0:
        raise ValueError(
            f"patch_shape {tuple(patch_shape)} must be a multiple of "
            f"2^(num_levels-1) = {div} (num_levels = len(ch_mult) = {num_levels}); "
            f"got remainders ({py % div}, {px % div}). Choose a patch shape that is "
            f"divisible by {div} on both axes."
        )


# ──────────────────────────────────────────────────────────────────────────
# Patch extraction helper
# ──────────────────────────────────────────────────────────────────────────
def _extract_patches(field: torch.Tensor, origins, patch_y: int, patch_x: int):
    """
    Slice P×P windows from `field` at each (y0, x0) in `origins` and stack them on
    the batch dim, patch-major / batch-minor: the returned index is `p*B + b`
    (patch `p`, sample `b`). All patchers, the global-PE gather and the fuse use
    this same ordering so residual / conditioning / PE stay aligned.

    `field` must already be padded (grid) or have origins guaranteed in-bounds
    (random). Returns (patch_num*B, C, patch_y, patch_x).
    """
    patches = [
        field[:, :, y0 : y0 + patch_y, x0 : x0 + patch_x] for (y0, x0) in origins
    ]
    return torch.cat(patches, dim=0)


def _axis_plan(size: int, patch: int, overlap: int, boundary: int):
    """
    Per-axis tiling plan. When `patch >= size` the axis is not patched (single
    window, no reflection pad, no boundary crop). Otherwise: stride =
    patch-overlap-boundary, `ceil` patch count, `boundary` reflection pad on the
    leading edge and whatever the trailing edge needs to make the grid even.

    Returns dict(num, stride, origins, pad_before, pad_after, boundary_eff).
    """
    if patch >= size:
        return {
            "num": 1,
            "stride": patch,
            "origins": [0],
            "pad_before": 0,
            "pad_after": 0,
            "boundary_eff": 0,
        }
    stride = patch - overlap - boundary
    if stride <= 0:
        raise ValueError(
            f"patch ({patch}) - overlap ({overlap}) - boundary ({boundary}) = "
            f"{stride} must be > 0."
        )
    num = math.ceil(size / stride)
    pad_before = boundary
    padded = (num - 1) * stride + patch
    pad_after = padded - pad_before - size
    origins = [i * stride for i in range(num)]
    return {
        "num": num,
        "stride": stride,
        "origins": origins,
        "pad_before": pad_before,
        "pad_after": pad_after,
        "boundary_eff": boundary,
    }


# ──────────────────────────────────────────────────────────────────────────
# Deterministic grid patcher (inference / regressor tiling)
# ──────────────────────────────────────────────────────────────────────────
class GridPatching2D:
    """
    Regular full-cover patch grid with reflection padding and an
    accumulate-and-average `fuse` that drops the outer `boundary_pix` ring of each
    interior patch (spec §3–§4).

    `fuse(apply(x)) == x` on the original domain: every patch is an exact crop of
    the same (padded) field, so overlapping pixels average identical values and,
    after the padding is cropped away, the full field is recovered. When
    `patch_shape >= img_shape` the patcher degenerates to a single full-domain
    window and both `apply` and `fuse` are the identity.
    """

    def __init__(self, img_shape, patch_shape, overlap_pix: int = 4, boundary_pix: int = 2):
        if overlap_pix < boundary_pix:
            raise ValueError(
                f"overlap_pix ({overlap_pix}) must be >= boundary_pix "
                f"({boundary_pix}) so kept interiors still overlap (no seams)."
            )
        self.H, self.W = int(img_shape[0]), int(img_shape[1])
        self.Py, self.Px = int(patch_shape[0]), int(patch_shape[1])
        if self.Py > self.H or self.Px > self.W:
            raise ValueError(
                f"patch_shape {(self.Py, self.Px)} must be <= img_shape "
                f"{(self.H, self.W)} on both axes."
            )
        self.overlap = overlap_pix
        self.boundary = boundary_pix

        py = _axis_plan(self.H, self.Py, overlap_pix, boundary_pix)
        px = _axis_plan(self.W, self.Px, overlap_pix, boundary_pix)
        self._py, self._px = py, px
        self.by, self.bx = py["boundary_eff"], px["boundary_eff"]
        self.pad = (px["pad_before"], px["pad_after"], py["pad_before"], py["pad_after"])
        self.padded_H = py["pad_before"] + self.H + py["pad_after"]
        self.padded_W = px["pad_before"] + self.W + px["pad_after"]
        self.origins = [(y0, x0) for y0 in py["origins"] for x0 in px["origins"]]
        self.patch_num = len(self.origins)
        self.active = self.Py < self.H or self.Px < self.W

    # -- origins API (parity with RandomPatching2D) --
    def new_origins(self, *args, **kwargs) -> None:
        """No-op: grid origins are fixed at construction."""
        return None

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad == (0, 0, 0, 0):
            return x
        return F.pad(x, self.pad, mode="reflect")

    def extract(self, field: torch.Tensor) -> torch.Tensor:
        """Reflection-pad `field` and slice all grid patches. (patch_num*B, C, Py, Px)."""
        return _extract_patches(self._pad(field), self.origins, self.Py, self.Px)

    # `apply` alias mirrors the Modulus name; keeps call-sites readable.
    def apply(self, field: torch.Tensor, additional_input: torch.Tensor = None):
        patches = self.extract(field)
        if additional_input is not None:
            add = additional_input.repeat(self.patch_num, 1, 1, 1)
            patches = torch.cat([patches, add], dim=1)
        return patches

    def fuse(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Accumulate-and-average interiors back to the full domain (spec §4). The
        outer `boundary_pix` ring of each patch is dropped; the remaining interior
        is summed into a padded accumulator and divided by the coverage count,
        then the reflection padding is cropped off.
        """
        B = patches.shape[0] // self.patch_num
        C = patches.shape[1]
        if not self.active:
            return patches  # single full-domain window
        dev, dt = patches.device, patches.dtype
        out = torch.zeros(B, C, self.padded_H, self.padded_W, device=dev, dtype=dt)
        cnt = torch.zeros(1, 1, self.padded_H, self.padded_W, device=dev, dtype=dt)
        by, bx = self.by, self.bx
        for p, (y0, x0) in enumerate(self.origins):
            blk = patches[p * B : (p + 1) * B]  # (B, C, Py, Px)
            ys, ye = y0 + by, y0 + self.Py - by
            xs, xe = x0 + bx, x0 + self.Px - bx
            out[:, :, ys:ye, xs:xe] += blk[:, :, by : self.Py - by, bx : self.Px - bx]
            cnt[:, :, ys:ye, xs:xe] += 1.0
        out = out / cnt.clamp(min=1.0)  # clamp: uncovered padding rim (cropped below)
        pl, _, pt, _ = self.pad
        return out[:, :, pt : pt + self.H, pl : pl + self.W]


# ──────────────────────────────────────────────────────────────────────────
# Random patcher (training)
# ──────────────────────────────────────────────────────────────────────────
class RandomPatching2D:
    """
    Extract `patch_num` random P×P patches per step (shared across the batch),
    mirroring CorrDiff's patched training. No reflection pad (origins are sampled
    fully in-bounds) and no fuse (the diffusion loss is computed directly on
    patches). Call `new_origins(H, W, device)` once per step, then `extract` each
    of the aligned full-domain tensors (residual, conditioning, PE).
    """

    def __init__(self, patch_shape, patch_num: int = 8):
        self.Py, self.Px = int(patch_shape[0]), int(patch_shape[1])
        self.patch_num = int(patch_num)
        self.origins = None

    def new_origins(self, H: int, W: int, device, generator=None) -> None:
        if self.Py > H or self.Px > W:
            raise ValueError(
                f"patch_shape {(self.Py, self.Px)} must be <= img_shape {(H, W)}."
            )
        max_y, max_x = H - self.Py, W - self.Px
        ys = torch.randint(0, max_y + 1, (self.patch_num,), generator=generator)
        xs = torch.randint(0, max_x + 1, (self.patch_num,), generator=generator)
        self.origins = [(int(y), int(x)) for y, x in zip(ys, xs)]

    def extract(self, field: torch.Tensor) -> torch.Tensor:
        if self.origins is None:
            raise RuntimeError("call new_origins(H, W, device) before extract().")
        return _extract_patches(field, self.origins, self.Py, self.Px)


# ──────────────────────────────────────────────────────────────────────────
# Conditioning assembly (shared by train + inference, diffusion + regressor)
# ──────────────────────────────────────────────────────────────────────────
def assemble_cond_patches(patcher, cond_local_full, global_source_full, K: int):
    """
    Build the per-patch conditioning at the current origins (call
    `patcher.new_origins(...)` first for the random patcher; no-op for the grid):

        cond_high = [ local crops of `cond_local_full` | global_lr thumbnail ]
        pos_embd  = per-patch crop of the full-domain spatial PE (or None if K==0)

    `global_lr` is `global_source_full` bilinearly resized to (Py, Px) and appended
    (same thumbnail on every patch). Returns (cond_high_patches, pos_embd_patches).
    """
    B, _, H, W = cond_local_full.shape
    dev, dt = cond_local_full.device, cond_local_full.dtype

    global_lr = F.interpolate(
        global_source_full, size=(patcher.Py, patcher.Px),
        mode="bilinear", align_corners=False,
    )
    cond_high = patcher.apply(cond_local_full, additional_input=global_lr) \
        if hasattr(patcher, "apply") else torch.cat(
            [patcher.extract(cond_local_full),
             global_lr.repeat(patcher.patch_num, 1, 1, 1)], dim=1)

    pos_embd = None
    if K > 0:
        pe_full = build_spatial_pe(H, W, K, dev, dt).expand(B, -1, -1, -1)
        pos_embd = patcher.extract(pe_full)
    return cond_high, pos_embd


def build_train_patcher(cfg: dict, backbone_kwargs: dict):
    """
    Build the training-time `RandomPatching2D` from a `patching:` YAML block and
    validate the parts of the backbone layout that don't need runtime channel
    counts (cond_low_channels, spatial_pe_freqs, patch_shape divisibility). The
    cond_high_channel count is checked later, once the first batch reveals
    C_x/C_y/C_f, via `validate_patched_cond_high`. Shared by `trainer_resdiff`
    and `trainer_song_unet_det`. Returns (patcher, K, patch_cfg_dict).
    """
    py, px = int(cfg["patch_shape_y"]), int(cfg["patch_shape_x"])
    overlap = int(cfg.get("overlap_pix", 4))
    boundary = int(cfg.get("boundary_pix", 2))
    patch_num = int(cfg.get("patch_num", 8))

    if int(backbone_kwargs.get("cond_low_channels", 0)) != 0:
        raise ValueError(
            "patched runs require the backbone's cond_low_channels: 0 — all "
            "conditioning is delivered at the HR patch grid via cond_high."
        )
    K = int(backbone_kwargs.get("spatial_pe_freqs", 0))
    if K < 1:
        raise ValueError(
            "patched runs require spatial_pe_freqs >= 1 so each patch receives a "
            "GLOBAL positional embedding (its location in the full domain)."
        )
    validate_patch_shape((py, px), len(backbone_kwargs.get("ch_mult", (1,))))

    patcher = RandomPatching2D((py, px), patch_num=patch_num)
    patch_cfg = {
        "enabled": True,
        "patch_shape_y": py, "patch_shape_x": px,
        "overlap_pix": overlap, "boundary_pix": boundary,
        "patch_num": patch_num, "K": K,
    }
    return patcher, K, patch_cfg


def build_grid_patcher(patch_cfg: dict, img_shape):
    """Reconstruct the inference-time `GridPatching2D` from a persisted patch_cfg
    (as stored in run metadata by `build_train_patcher`) and the full-domain
    (H, W). Returns (patcher, K)."""
    patcher = GridPatching2D(
        img_shape,
        (patch_cfg["patch_shape_y"], patch_cfg["patch_shape_x"]),
        overlap_pix=patch_cfg.get("overlap_pix", 4),
        boundary_pix=patch_cfg.get("boundary_pix", 2),
    )
    return patcher, int(patch_cfg["K"])


def validate_patched_cond_high(cond_high_channels: int, C_x: int, C_y: int,
                               C_f: int, stage: str) -> None:
    """Assert the backbone's declared cond_high_channels matches the assembled
    patched conditioning `[local crops | global_lr thumbnail]`. Called on the
    first training batch (and at inference) once C_x/C_y/C_f are known."""
    if stage == "diffusion":
        expected, layout = C_y + 2 * C_x, f"C_y({C_y}) + 2*C_x({C_x})"
    elif stage == "regressor":
        expected, layout = 2 * C_x + C_f, f"2*C_x({C_x}) + C_f({C_f})"
    else:
        raise ValueError(f"unknown stage '{stage}'")
    if int(cond_high_channels) != expected:
        raise ValueError(
            f"patched {stage}: backbone cond_high_channels must be {expected} "
            f"(= {layout}), got {cond_high_channels}. The patched conditioning is "
            f"[local crops | global_lr thumbnail]."
        )


def run_regressor_patched(model, cond_low_hr, forcing_hr, patcher: GridPatching2D,
                          K: int, C_y: int) -> torch.Tensor:
    """
    Deterministic grid apply → forward → fuse for a patched SongUNet regressor,
    reused by `downscaler_song_unet_det`, `downscaler_resdiff` and
    `pydataset_resdiff` so the three sites stay identical.

    Parameters
    ----------
    model       : patched SongUNet regressor (cond_low_channels=0).
    cond_low_hr : (B, C_x, H, W) low-res predictors ALREADY upsampled to the HR
                  grid and normalized.
    forcing_hr  : (B, C_f, H, W) HR forcing (e.g. orography) or None.
    patcher     : GridPatching2D matching the regressor's training geometry.
    K           : spatial_pe_freqs of the regressor backbone.
    C_y         : number of predictand channels (the model's `in_channels`).

    Returns
    -------
    (B, C_y, H, W) fused mean prediction.
    """
    B, _, H, W = cond_low_hr.shape
    dev, dt = cond_low_hr.device, cond_low_hr.dtype
    patcher.new_origins(H, W, dev)

    cond_local = cond_low_hr if forcing_hr is None else torch.cat(
        [cond_low_hr, forcing_hr], dim=1)
    cond_high, pos_embd = assemble_cond_patches(patcher, cond_local, cond_low_hr, K)

    Pn = patcher.patch_num
    x_in = torch.zeros(Pn * B, C_y, patcher.Py, patcher.Px, device=dev, dtype=dt)
    t = torch.zeros(Pn * B, device=dev)
    out_patches = model(x=x_in, t=t, cond_low=None, cond_high=cond_high, pos_embd=pos_embd)
    return patcher.fuse(out_patches)
