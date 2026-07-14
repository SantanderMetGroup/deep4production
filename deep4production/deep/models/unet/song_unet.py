"""
SongUNet: generalized NCSN++ / Song et al. U-Net backbone for diffusion models.

Derived from Song et al. 2021 (NCSN++) and Addison et al. 2024 (CPMGEM),
generalized to accept two conditioning streams at different spatial resolutions:

  - cond_low  : low-resolution conditioning (upsampled internally, FIR or nearest)
  - cond_high : high-resolution conditioning (already at target spatial size)

This makes the same backbone reusable by different diffusion frameworks:

  - CPMGEM-style  (sub-VP SDE, low-res conditioning only)      cond_high_channels=0
  - CorrDiff-style (EDM SDE, low-res inputs + regression mean) cond_high_channels>0

The SDE framework (sub-VP vs EDM), the preconditioning (c_in, c_skip, c_out),
the noise schedule, and the loss weighting all live *outside* this module, in
trainers / loss functions / optional preconditioner wrappers (see
deep4production/deep/models/diffusion/edm_precond.py for EDM).

The scalar `t` passed to forward() is treated as a generic noise label and is
embedded via sinusoidal PE + MLP. The trainer decides its semantics:
  - sub-VP : t ∈ (0, 1] (continuous diffusion time)
  - EDM    : c_noise = 0.25 · log(σ_t)

Architecture features (Song et al. 2021 / Addison et al. 2024):
  - BigGAN-style ResBlocks with additive time conditioning
  - DDPM++ attention blocks at user-specified levels
  - FIR-filtered up/downsampling (kernel [1, 3, 3, 1])
  - Skip connections rescaled by 1/√2
  - Progressive input (input pyramid added at each encoder level)
  - Zero-initialised output conv

Author:
    Jorge Baño-Medina
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from deep4production.deep.models.diffusion.patching import build_spatial_pe


# ──────────────────────────────────────────────────────────────────────────
# FIR up/downsampling
# ──────────────────────────────────────────────────────────────────────────


def _fir_kernel_2d(kernel, device, dtype):
    k = torch.tensor(kernel, dtype=torch.float32, device=device)
    k = torch.outer(k, k)
    k = k / k.sum()
    return k.to(dtype)


def fir_downsample(x: torch.Tensor, kernel=(1, 3, 3, 1)) -> torch.Tensor:
    C = x.shape[1]
    k = _fir_kernel_2d(kernel, x.device, x.dtype)
    w = k.view(1, 1, *k.shape).expand(C, 1, -1, -1)
    pad = (k.shape[0] - 2) // 2
    x = F.pad(x, [pad, pad, pad, pad])
    return F.conv2d(x, w, stride=2, groups=C)


def fir_upsample(x: torch.Tensor, kernel=(1, 3, 3, 1)) -> torch.Tensor:
    C = x.shape[1]
    k = _fir_kernel_2d(kernel, x.device, x.dtype) * 4.0
    w = k.view(1, 1, *k.shape).expand(C, 1, -1, -1)
    kH = k.shape[0]
    pad = (kH - 2) // 2
    return F.conv_transpose2d(x, w, stride=2, padding=pad, groups=C)


# ──────────────────────────────────────────────────────────────────────────
# Noise-label embedding
# ──────────────────────────────────────────────────────────────────────────


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding of a 1-D scalar per sample."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / (half - 1)
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class NoiseEmbedMLP(nn.Module):
    """PE → 2-layer MLP. Consumes any scalar noise label."""

    def __init__(self, nf: int) -> None:
        super().__init__()
        emb_dim = nf * 4
        self.net = nn.Sequential(
            PositionalEmbedding(nf),
            nn.Linear(nf, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


# ──────────────────────────────────────────────────────────────────────────
# BigGAN ResBlock and attention
# ──────────────────────────────────────────────────────────────────────────


def _num_groups(channels: int) -> int:
    return min(channels // 4, 32)


def _init_near_zero_(weight: torch.Tensor, scale: float = 1e-10) -> None:
    """In-place variance-scaling init matching score_sde_pytorch's
    ``default_init(scale=0)`` (mlde / Song et al. NCSN++).

    Equivalent JAX/Score-SDE recipe: uniform with variance ``scale/fan_avg``,
    where ``scale = 1e-10`` is the clamp for "zero" initialization. The
    resulting weights are vanishingly small (e.g. O(1e-7) for a 3×3 conv with
    128 channels) but **non-zero**, which is what lets gradients propagate
    through a chain of "zero-init" layers from the very first optimizer step.
    Biases are zeroed separately by the caller.
    """
    if weight.dim() != 4:
        raise ValueError(f"expected 4D Conv2d weight, got {weight.dim()}D")
    out_ch, in_ch, kH, kW = weight.shape
    rf = kH * kW
    fan_avg = (in_ch + out_ch) * rf / 2.0
    bound = math.sqrt(3.0 * scale / fan_avg)
    nn.init.uniform_(weight, -bound, bound)


class ResBlock(nn.Module):
    """BigGAN-style ResBlock with additive time conditioning."""

    def __init__(
        self,
        act: nn.Module,
        in_ch: int,
        out_ch: int,
        emb_dim: int,
        up: bool = False,
        down: bool = False,
        dropout: float = 0.1,
        fir: bool = True,
        fir_kernel: tuple = (1, 3, 3, 1),
        skip_rescale: bool = True,
    ) -> None:
        super().__init__()
        assert not (up and down), "up and down are mutually exclusive"

        self.act = act
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.skip_rescale = skip_rescale

        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch, eps=1e-6)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.dense = nn.Linear(emb_dim, out_ch)
        nn.init.zeros_(self.dense.bias)

        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        _init_near_zero_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        if in_ch != out_ch or up or down:
            self.skip_proj = nn.Conv2d(in_ch, out_ch, 1)

    def _resample(self, x: torch.Tensor) -> torch.Tensor:
        if self.up:
            return (
                fir_upsample(x, self.fir_kernel)
                if self.fir
                else F.interpolate(x, scale_factor=2, mode="nearest")
            )
        if self.down:
            return (
                fir_downsample(x, self.fir_kernel)
                if self.fir
                else F.avg_pool2d(x, 2, stride=2)
            )
        return x

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self._resample(h)
        x = self._resample(x)
        h = self.conv1(h)
        h = h + self.dense(self.act(temb))[:, :, None, None]
        h = self.dropout(self.act(self.norm2(h)))
        h = self.conv2(h)
        if hasattr(self, "skip_proj"):
            x = self.skip_proj(x)
        return (x + h) / math.sqrt(2.0) if self.skip_rescale else x + h


class AttnBlock(nn.Module):
    """Channel self-attention with GroupNorm pre-norm and residual."""

    def __init__(self, channels: int, skip_rescale: bool = True) -> None:
        super().__init__()
        self.skip_rescale = skip_rescale
        self.norm = nn.GroupNorm(_num_groups(channels), channels, eps=1e-6)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        _init_near_zero_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).view(B, C, -1)
        k = self.k(h).view(B, C, -1)
        v = self.v(h).view(B, C, -1)
        w = torch.bmm(q.permute(0, 2, 1), k) * (C**-0.5)
        w = w.softmax(dim=-1)
        h = torch.bmm(v, w.permute(0, 2, 1)).view(B, C, H, W)
        out = x + self.proj(h)
        return out / math.sqrt(2.0) if self.skip_rescale else out


# ──────────────────────────────────────────────────────────────────────────
# Progressive-input pyramid downsampler
# ──────────────────────────────────────────────────────────────────────────


class PyramidDown(nn.Module):
    """FIR-filtered (or avg-pool) spatial downsample followed by a 3×3 projection.

    Mirrors ``layerspp.Downsample(with_conv=True, fir=True)`` as used by
    ``cNCSNpp`` for ``progressive_input='residual'``: the pyramid is carried
    level-by-level via a learnable conv, not just blind downsampling.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        fir: bool = True,
        fir_kernel: tuple = (1, 3, 3, 1),
    ) -> None:
        super().__init__()
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (
            fir_downsample(x, self.fir_kernel)
            if self.fir
            else F.avg_pool2d(x, 2, stride=2)
        )
        return self.conv(x)


# ──────────────────────────────────────────────────────────────────────────
# SongUNet
# ──────────────────────────────────────────────────────────────────────────


class SongUNet(nn.Module):
    """
    Generalized Song et al. / NCSN++ U-Net backbone.

    Forward: (x, t, cond_low=None, cond_high=None) → y_hat, where:
      - x         : (B, in_channels, H, W) — noisy input
      - t         : (B,) or (B, 1, 1, 1) — scalar noise label per sample
      - cond_low  : (B, cond_low_channels, Hc, Wc) — low-res conditioning,
                    upsampled internally to (H, W)
      - cond_high : (B, cond_high_channels, H, W) — already at target resolution

    The output has the same shape as `x` and represents whatever the outer
    training objective requires (score / noise ε for sub-VP; raw F_θ for an
    EDM preconditioner wrapper).

    Parameters
    ----------
    in_channels : int
        Channels of the noisy input `x`.
    cond_low_channels : int
        Channels of the low-resolution conditioning. 0 disables this stream.
    cond_high_channels : int
        Channels of the high-resolution conditioning. 0 disables this stream.
    nf : int
        Base channel width.
    ch_mult : tuple[int]
        Per-level channel multipliers.
    num_res_blocks : int
        ResBlocks per encoder/decoder level.
    attn_at_levels : tuple[int]
        Encoder levels (0-indexed) where attention is applied.
    dropout : float
    fir : bool
        Use FIR-filtered up/downsampling inside ResBlocks.
    fir_kernel : tuple[int]
        1-D FIR prototype.
    skip_rescale : bool
        Divide residual branches by √2.
    progressive_input : bool
        Add a projected, downsampled copy of the raw input stack at each
        encoder level (NCSN++ "residual" input mode).
    cond_upsample : {"fir", "nearest"}
        How to upsample cond_low to the target resolution.
    spatial_pe_freqs : int
        Number of spatial positional-embedding frequency bands per axis.
        0 disables. K > 0 adds 4·K channels at the input stack: for each
        frequency index k ∈ [0, K−1] the channels are
            sin(2^k · 2π · y/H), cos(2^k · 2π · y/H),
            sin(2^k · 2π · x/W), cos(2^k · 2π · x/W).
        K = 1 matches CorrDiff (Mardani et al. 2023); higher K adds
        multi-scale spatial detail (NeRF-style).
    """

    def __init__(
        self,
        in_channels: int,
        cond_low_channels: int = 0,
        cond_high_channels: int = 0,
        nf: int = 128,
        ch_mult: tuple = (1, 2, 2, 2),
        num_res_blocks: int = 4,
        attn_at_levels: tuple = (2, 3),
        dropout: float = 0.1,
        fir: bool = True,
        fir_kernel: tuple = (1, 3, 3, 1),
        skip_rescale: bool = True,
        progressive_input: bool = True,
        cond_upsample: str = "fir",
        spatial_pe_freqs: int = 0,
    ) -> None:
        super().__init__()

        assert cond_upsample in (
            "fir",
            "nearest",
        ), f"cond_upsample must be 'fir' or 'nearest', got '{cond_upsample}'"

        act = nn.SiLU()
        emb_dim = nf * 4
        num_levels = len(ch_mult)
        chs = [nf * m for m in ch_mult]

        self.in_channels = in_channels
        self.cond_low_channels = cond_low_channels
        self.cond_high_channels = cond_high_channels
        self.act = act
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.progressive_input = progressive_input
        self.cond_upsample = cond_upsample
        self.skip_rescale = skip_rescale

        # Spatial PE: K frequencies per axis → 4·K extra channels (0 disables).
        # The tensor is batch-independent and a pure function of (H, W); we
        # compute it lazily on the first forward and cache it, so it costs
        # nothing per step beyond a single torch.cat.
        assert spatial_pe_freqs >= 0, "spatial_pe_freqs must be >= 0"
        self.spatial_pe_freqs = spatial_pe_freqs
        self.spatial_pe_channels = 4 * spatial_pe_freqs
        self._spatial_pe_cache = None  # populated on first forward()

        # Noise-label embedding
        self.noise_embed = NoiseEmbedMLP(nf)

        # Input projection: noisy + upsampled cond_low + cond_high + spatial PE
        net_in_ch = (
            in_channels
            + cond_low_channels
            + cond_high_channels
            + self.spatial_pe_channels
        )
        self.in_conv = nn.Conv2d(net_in_ch, nf, 3, padding=1)

        # Encoder — per-resblock skips + post-downsample skip (matches mlde).
        # Each encoder level contributes num_res_blocks "resblock" skips plus,
        # for non-terminal levels, one "post-down" skip. Together with the
        # initial post-in_conv skip this yields num_levels × (num_res_blocks+1)
        # skips total, exactly consumed by the decoder.
        self.enc_resnets = nn.ModuleList()
        self.enc_attns = nn.ModuleList()
        self.enc_downs = nn.ModuleList()
        self.prog_downs = nn.ModuleList()  # pyramid FIR-down + conv per level

        in_ch = nf
        pyramid_ch = net_in_ch
        for level in range(num_levels):
            out_ch = chs[level]
            use_attn = level in attn_at_levels
            lvl_res = nn.ModuleList()
            lvl_atn = nn.ModuleList()
            for _ in range(num_res_blocks):
                lvl_res.append(
                    ResBlock(
                        act,
                        in_ch,
                        out_ch,
                        emb_dim,
                        dropout=dropout,
                        fir=fir,
                        fir_kernel=fir_kernel,
                        skip_rescale=skip_rescale,
                    )
                )
                lvl_atn.append(
                    AttnBlock(out_ch, skip_rescale=skip_rescale) if use_attn else None
                )
                in_ch = out_ch
            self.enc_resnets.append(lvl_res)
            self.enc_attns.append(lvl_atn)

            if level < num_levels - 1:
                next_ch = chs[level + 1]
                self.enc_downs.append(
                    ResBlock(
                        act,
                        in_ch,
                        next_ch,
                        emb_dim,
                        down=True,
                        dropout=dropout,
                        fir=fir,
                        fir_kernel=fir_kernel,
                        skip_rescale=skip_rescale,
                    )
                )
                if progressive_input:
                    self.prog_downs.append(
                        PyramidDown(pyramid_ch, next_ch, fir=fir, fir_kernel=fir_kernel)
                    )
                    pyramid_ch = next_ch
                else:
                    self.prog_downs.append(None)
                in_ch = next_ch
            else:
                self.enc_downs.append(None)
                self.prog_downs.append(None)

        # Bottleneck
        ch = chs[-1]
        self.mid_res1 = ResBlock(
            act,
            ch,
            ch,
            emb_dim,
            dropout=dropout,
            fir=fir,
            fir_kernel=fir_kernel,
            skip_rescale=skip_rescale,
        )
        self.mid_attn = AttnBlock(ch, skip_rescale=skip_rescale)
        self.mid_res2 = ResBlock(
            act,
            ch,
            ch,
            emb_dim,
            dropout=dropout,
            fir=fir,
            fir_kernel=fir_kernel,
            skip_rescale=skip_rescale,
        )

        # Decoder — num_res_blocks+1 ResBlocks per level, each consuming a
        # fresh skip from the encoder via channel-wise concat (matches mlde).
        # One AttnBlock per attn level, applied after all resblocks.
        self.dec_resnets = nn.ModuleList()
        self.dec_attns = nn.ModuleList()  # single AttnBlock or None, one per level
        self.dec_ups = nn.ModuleList()

        # At each decoder level, all popped skips share the level's channel
        # count (the "down"-skip from level l-1 was projected to chs[l] by the
        # encoder's down-resblock). So block i's in_ch = h_ch + chs[level],
        # where h_ch = ch (running) for block 0 and = chs[level] after.
        for level in reversed(range(num_levels)):
            out_ch = chs[level]
            skip_ch = chs[level]
            use_attn = level in attn_at_levels
            lvl_res = nn.ModuleList()

            for i in range(num_res_blocks + 1):
                in_ch_blk = (ch if i == 0 else out_ch) + skip_ch
                lvl_res.append(
                    ResBlock(
                        act,
                        in_ch_blk,
                        out_ch,
                        emb_dim,
                        dropout=dropout,
                        fir=fir,
                        fir_kernel=fir_kernel,
                        skip_rescale=skip_rescale,
                    )
                )
            self.dec_resnets.append(lvl_res)
            self.dec_attns.append(
                AttnBlock(out_ch, skip_rescale=skip_rescale) if use_attn else None
            )
            ch = out_ch

            if level > 0:
                self.dec_ups.append(
                    ResBlock(
                        act,
                        ch,
                        ch,
                        emb_dim,
                        up=True,
                        dropout=dropout,
                        fir=fir,
                        fir_kernel=fir_kernel,
                        skip_rescale=skip_rescale,
                    )
                )
            else:
                self.dec_ups.append(None)

        # Output head — two 3×3 convs through a net_in_ch bottleneck, matching
        # mlde. Both convs are near-zero-init (variance ≈ 1e-10/fan_avg) so the
        # initial network output is ~0 (stable start) yet gradients still flow
        # through the chain from the first optimizer step.
        self.out_norm = nn.GroupNorm(_num_groups(ch), ch, eps=1e-6)
        self.out_conv_mid = nn.Conv2d(ch, net_in_ch, 3, padding=1)
        self.out_conv = nn.Conv2d(net_in_ch, in_channels, 3, padding=1)
        _init_near_zero_(self.out_conv_mid.weight)
        nn.init.zeros_(self.out_conv_mid.bias)
        _init_near_zero_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    # ---------------------------------------------------------------------
    def _build_spatial_pe(self, H: int, W: int, device, dtype) -> torch.Tensor:
        """Delegate to the shared `build_spatial_pe` so this local PE and the
        GLOBAL PE gathered per patch by the patchers are byte-identical."""
        return build_spatial_pe(H, W, self.spatial_pe_freqs, device, dtype)

    # ---------------------------------------------------------------------
    def _get_spatial_pe(self, target_hw: tuple, device, dtype) -> torch.Tensor:
        """Return the cached spatial PE; rebuild on first call or when the
        target shape / device / dtype changes. Not a registered buffer —
        it's a pure function of (H, W) so we don't persist it in state_dict."""
        H, W = target_hw
        cache = self._spatial_pe_cache
        if (
            cache is None
            or cache.shape[-2:] != (H, W)
            or cache.device != device
            or cache.dtype != dtype
        ):
            self._spatial_pe_cache = self._build_spatial_pe(H, W, device, dtype)
        return self._spatial_pe_cache

    # ---------------------------------------------------------------------
    def _upsample_cond_low(
        self, cond_low: torch.Tensor, target_hw: tuple
    ) -> torch.Tensor:
        if self.cond_upsample == "fir":
            up = fir_upsample(cond_low, self.fir_kernel)
        else:
            up = F.interpolate(cond_low, size=target_hw, mode="nearest")
        if up.shape[-2:] != target_hw:
            up = F.interpolate(up, size=target_hw, mode="bilinear", align_corners=False)
        return up

    # ---------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_low: torch.Tensor = None,
        cond_high: torch.Tensor = None,
        pos_embd: torch.Tensor = None,
    ) -> torch.Tensor:
        # 1. Build the channel-stacked input
        parts = [x]
        if self.cond_low_channels > 0:
            assert cond_low is not None, "cond_low_channels>0 but cond_low is None"
            parts.append(self._upsample_cond_low(cond_low, x.shape[-2:]))
        else:
            assert cond_low is None, "cond_low passed but cond_low_channels=0"
        if self.cond_high_channels > 0:
            assert cond_high is not None, "cond_high_channels>0 but cond_high is None"
            parts.append(cond_high)
        else:
            assert cond_high is None, "cond_high passed but cond_high_channels=0"
        if self.spatial_pe_freqs > 0:
            # Patch-based diffusion supplies a GLOBAL positional embedding gathered
            # per patch (each patch's slice of the full-domain PE); when given we
            # use it verbatim instead of the locally-built PE, which would wrongly
            # make every patch think it spans the whole [0,1] domain.
            if pos_embd is not None:
                assert pos_embd.shape[1] == self.spatial_pe_channels, (
                    f"pos_embd has {pos_embd.shape[1]} channels, expected "
                    f"spatial_pe_channels={self.spatial_pe_channels}"
                )
                parts.append(pos_embd)
            else:
                pe = self._get_spatial_pe(x.shape[-2:], x.device, x.dtype)
                parts.append(pe.expand(x.shape[0], -1, -1, -1))
        else:
            assert pos_embd is None, "pos_embd passed but spatial_pe_freqs=0"
        x_cat = torch.cat(parts, dim=1) if len(parts) > 1 else x

        # 2. Noise-label embedding (accept (B,) or (B,1,1,1))
        if t.dim() > 1:
            t = t.view(t.shape[0])
        t_emb = self.noise_embed(t)

        # 3. Input conv, push initial skip
        h = self.in_conv(x_cat)
        skips = [h]

        # 4. Encoder
        x_pyramid = x_cat
        for level in range(len(self.enc_resnets)):
            for resblock, attn in zip(self.enc_resnets[level], self.enc_attns[level]):
                h = resblock(h, t_emb)
                if attn is not None:
                    h = attn(h)
                skips.append(h)

            if self.enc_downs[level] is not None:
                h = self.enc_downs[level](h, t_emb)
                if self.progressive_input:
                    # Pyramid is carried via a learnable FIR-down + 3×3 conv,
                    # combined with h at the *downsampled* resolution, then
                    # replaces h so the next level sees the mixed signal.
                    x_pyramid = self.prog_downs[level](x_pyramid)
                    h = (
                        (x_pyramid + h) / math.sqrt(2.0)
                        if self.skip_rescale
                        else (x_pyramid + h)
                    )
                    x_pyramid = h
                skips.append(h)

        # 5. Bottleneck
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        # 6. Decoder — num_res_blocks+1 ResBlocks per level, each concatenating
        # a fresh encoder skip. Single attention per attn-level, after all
        # resblocks. Matches cNCSNpp exactly.
        for lvl_res, lvl_atn, upsampler in zip(
            self.dec_resnets, self.dec_attns, self.dec_ups
        ):
            for resblock in lvl_res:
                skip = skips.pop()
                h = resblock(torch.cat([h, skip], dim=1), t_emb)
            if lvl_atn is not None:
                h = lvl_atn(h)
            if upsampler is not None:
                h = upsampler(h, t_emb)
        assert not skips, f"unconsumed encoder skips: {len(skips)}"

        # 7. Output — GN → act → Conv3x3(ch, net_in_ch) → Conv3x3(net_in_ch, in_channels).
        h = self.act(self.out_norm(h))
        h = self.out_conv_mid(h)
        return self.out_conv(h)
