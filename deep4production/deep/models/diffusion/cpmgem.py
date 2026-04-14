"""
CPMGEM: Score/noise-prediction network for climate downscaling via sub-VP SDE.

Architecture based on:
  Addison et al. (2024) "Machine learning emulation of precipitation from
  km-scale UK regional climate simulations using a diffusion model"
  arXiv:2407.14158

Backbone follows Song et al. (2021) NCSN++ / ho2020 U-Net style:
  - Residual blocks with GroupNorm + SiLU + AdaGN time conditioning
  - Multi-head self-attention at user-specified depth levels
  - Sinusoidal timestep embedding passed through an MLP
  - Conditioning: low-res predictors are nearest-neighbour upsampled to the
    high-res target grid and concatenated channel-wise before the U-Net

Authors:
    Jorge Baño-Medina
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Time / noise-level embedding
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    """
    Maps scalar timesteps t ∈ [0,1] to a sinusoidal feature vector.

    Parameters
    ----------
    dim : int
        Output embedding dimensionality (must be even).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t : (B,)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
        )  # (half,)
        args = t[:, None].float() * freqs[None, :]  # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


class TimeEmbedMLP(nn.Module):
    """
    Projects sinusoidal time embeddings to the model's internal embedding dim
    via a two-layer MLP with SiLU activations.

    Parameters
    ----------
    sin_dim : int
        Dimensionality of the sinusoidal input embedding.
    emb_dim : int
        Output embedding dimensionality fed into ResBlocks.
    """

    def __init__(self, sin_dim: int, emb_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalEmbedding(sin_dim),
            nn.Linear(sin_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)  # (B, emb_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Core building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    Residual block with GroupNorm, SiLU activations, and Adaptive Group
    Normalisation (AdaGN) time conditioning (scale + shift on the first norm).

    Parameters
    ----------
    in_channels : int
    out_channels : int
    emb_dim : int
        Dimensionality of the time embedding vector.
    num_groups : int
        Number of groups for GroupNorm.
    dropout : float
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        num_groups: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1  = nn.GroupNorm(num_groups, in_channels)
        self.act1   = nn.SiLU()
        self.conv1  = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # AdaGN: time_emb → (scale, shift) applied after conv1
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels * 2),
        )

        self.norm2   = nn.GroupNorm(num_groups, out_channels)
        self.act2    = nn.SiLU()
        self.drop    = nn.Dropout(dropout)
        self.conv2   = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))

        # AdaGN: inject timestep via scale + shift
        scale_shift = self.time_proj(t_emb)[:, :, None, None]  # (B, 2C, 1, 1)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1.0 + scale) + shift

        h = self.drop(self.act2(self.norm2(h)))
        h = self.conv2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """
    Multi-head self-attention on spatial feature maps (NCSN++ style).
    Applies GroupNorm before attention and adds a residual connection.

    Parameters
    ----------
    channels : int
    num_heads : int
    num_groups : int
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        num_groups: int = 32,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + self.proj(h)


def _make_enc_level(
    in_ch: int,
    out_ch: int,
    emb_dim: int,
    num_res_blocks: int,
    use_attn: bool,
    num_groups: int,
    dropout: float,
) -> nn.ModuleList:
    """Build a list of ResBlock (+ optional AttentionBlock) for one encoder level."""
    blocks = nn.ModuleList()
    for i in range(num_res_blocks):
        blocks.append(ResBlock(in_ch if i == 0 else out_ch, out_ch, emb_dim, num_groups, dropout))
        if use_attn:
            blocks.append(AttentionBlock(out_ch, num_groups=num_groups))
    return blocks


def _make_dec_level(
    in_ch: int,       # = skip_ch * 2  (after upsampling + cat)
    out_ch: int,      # = skip_ch
    emb_dim: int,
    num_res_blocks: int,
    use_attn: bool,
    num_groups: int,
    dropout: float,
) -> nn.ModuleList:
    """Build a list of ResBlock (+ optional AttentionBlock) for one decoder level."""
    blocks = nn.ModuleList()
    for i in range(num_res_blocks):
        blocks.append(ResBlock(in_ch if i == 0 else out_ch, out_ch, emb_dim, num_groups, dropout))
        if use_attn:
            blocks.append(AttentionBlock(out_ch, num_groups=num_groups))
    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# CPMGEM
# ─────────────────────────────────────────────────────────────────────────────

class CPMGEM(nn.Module):
    """
    Score / noise-prediction U-Net for climate downscaling via sub-VP SDE.

    The forward call accepts the noisy high-resolution target y_t, the
    low-resolution conditioning field x_cond, and the normalised timestep t.
    x_cond is upsampled internally (nearest-neighbour) to match y_t's spatial
    dimensions before concatenation.

    Parameters
    ----------
    in_channels : int
        Number of channels in the high-resolution target (C_y).
    cond_channels : int
        Number of channels in the low-resolution predictor (C_x).
    base_channels : int
        Base channel width of the U-Net.  All levels are multiples of this.
        Default 128 reproduces the ~63 M parameter count of Addison et al.
        Use 64 for a lighter ~16 M version.
    channel_mults : tuple[int]
        Channel multipliers at each encoder level (coarsest → finest spatial
        resolution within the encoder).  Default (1, 2, 4, 8).
    num_res_blocks : int
        Number of ResBlocks per encoder / decoder level.
    attn_at_levels : tuple[int]
        0-indexed encoder levels at which self-attention is applied.
        Level 0 is the shallowest (full resolution), higher indices are deeper.
        Default (2, 3): attention at the two deepest levels.
    num_groups : int
        Number of groups for GroupNorm.  Must divide all channel counts.
    dropout : float
        Dropout probability inside ResBlocks.

    Notes
    -----
    U-Net layout for the defaults (base_channels=128, channel_mults=(1,2,4,8)):

      Input        : (B, C_y + C_x, H, W)
      Enc level 0  : 128 ch, H×W,   no attn  → skip0, then ↓2
      Enc level 1  : 256 ch, H/2×W/2, no attn → skip1, then ↓2
      Enc level 2  : 512 ch, H/4×W/4, attn   → skip2, then ↓2
      Enc level 3  : 1024 ch, H/8×W/8, attn  → feeds bottleneck directly
      Bottleneck   : 1024 ch, H/8×W/8, ResBlock + Attn + ResBlock
      Dec level 0  : ↑2 → concat skip2 → 1024 ch → 512 ch, attn
      Dec level 1  : ↑2 → concat skip1 → 512 ch  → 256 ch, no attn
      Dec level 2  : ↑2 → concat skip0 → 256 ch  → 128 ch, no attn
      Output       : 1×1 conv → (B, C_y, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        base_channels: int = 128,
        channel_mults: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_at_levels: tuple = (2, 3),
        num_groups: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Validate GroupNorm compatibility
        for mult in channel_mults:
            ch = base_channels * mult
            assert ch % num_groups == 0, (
                f"base_channels*mult={ch} must be divisible by num_groups={num_groups}"
            )

        num_levels = len(channel_mults)
        chs        = [base_channels * m for m in channel_mults]
        emb_dim    = base_channels * 4

        # ── Time embedding ────────────────────────────────────────────────────
        self.time_embed = TimeEmbedMLP(base_channels, emb_dim)

        # ── Input projection ──────────────────────────────────────────────────
        # Concatenated input has (in_channels + cond_channels) channels
        self.in_conv = nn.Conv2d(in_channels + cond_channels, base_channels, 3, padding=1)

        # ── Encoder ───────────────────────────────────────────────────────────
        # enc_blocks[i]     : ModuleList of ResBlock / AttentionBlock
        # downsamplers[i]   : strided conv for levels 0 … num_levels-2
        # Skips are stored for encoder levels 0 … num_levels-2 (before bottleneck)

        self.enc_blocks   = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        enc_skip_ch       = []   # channel count at each skip connection point

        in_ch = base_channels
        for level in range(num_levels):
            out_ch   = chs[level]
            use_attn = level in attn_at_levels
            self.enc_blocks.append(
                _make_enc_level(in_ch, out_ch, emb_dim, num_res_blocks, use_attn, num_groups, dropout)
            )
            if level < num_levels - 1:
                enc_skip_ch.append(out_ch)
                self.downsamplers.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            in_ch = out_ch

        # ch = chs[-1] at this point (output of deepest encoder level)
        ch = chs[-1]

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.mid_block1 = ResBlock(ch, ch, emb_dim, num_groups, dropout)
        self.mid_attn   = AttentionBlock(ch, num_groups=num_groups)
        self.mid_block2 = ResBlock(ch, ch, emb_dim, num_groups, dropout)

        # ── Decoder ───────────────────────────────────────────────────────────
        # Iterates from deepest skip (num_levels-2) back to shallowest (0).
        # Each level:  upsample (ch → skip_ch)  →  cat skip  →  ResBlocks (2*skip_ch → skip_ch)

        self.dec_blocks  = nn.ModuleList()
        self.upsamplers  = nn.ModuleList()

        for level in reversed(range(num_levels - 1)):
            skip_ch  = enc_skip_ch[level]
            use_attn = level in attn_at_levels
            self.upsamplers.append(nn.ConvTranspose2d(ch, skip_ch, 2, stride=2))
            self.dec_blocks.append(
                _make_dec_level(skip_ch * 2, skip_ch, emb_dim, num_res_blocks, use_attn, num_groups, dropout)
            )
            ch = skip_ch

        # ── Output ────────────────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(num_groups, ch)
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, in_channels, 1)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        y_t:    torch.Tensor,
        x_cond: torch.Tensor,
        t:      torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        y_t : (B, C_y, H_y, W_y)
            Noisy high-resolution target at diffusion step t.
        x_cond : (B, C_x, H_x, W_x)
            Low-resolution conditioning field; upsampled internally to H_y×W_y.
        t : (B,)
            Normalised diffusion timestep in [0, 1].

        Returns
        -------
        (B, C_y, H_y, W_y)
            Predicted noise ε.
        """
        # 1. Upsample conditioning to target spatial size
        x_up = F.interpolate(x_cond, size=y_t.shape[-2:], mode="nearest")
        h    = torch.cat([y_t, x_up], dim=1)   # (B, C_y + C_x, H_y, W_y)

        # 2. Time embedding
        t_emb = self.time_embed(t)              # (B, emb_dim)

        # 3. Input conv
        h = self.in_conv(h)

        # 4. Encoder — collect skip connections
        skips = []
        for level in range(len(self.enc_blocks)):
            for block in self.enc_blocks[level]:
                h = block(h, t_emb) if isinstance(block, ResBlock) else block(h)
            if level < len(self.enc_blocks) - 1:   # store skip before downsampling
                skips.append(h)
                h = self.downsamplers[level](h)

        # 5. Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # 6. Decoder — upsample, concat skip, process
        for i, (dec_level, upsampler) in enumerate(zip(self.dec_blocks, self.upsamplers)):
            h    = upsampler(h)
            skip = skips[-(i + 1)]
            h    = torch.cat([h, skip], dim=1)
            for block in dec_level:
                h = block(h, t_emb) if isinstance(block, ResBlock) else block(h)

        # 7. Output projection
        return self.out_conv(self.out_act(self.out_norm(h)))
