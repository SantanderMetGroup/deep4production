# Corrdiff reference site: https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/weather/corrdiff/README.html?utm_source=chatgpt.com#training-the-diffusion-model
# Corrdiff github: https://github.com/NVIDIA/physicsnemo/tree/main/examples/weather/corrdiff
# Corrdiff regressor UNET model (SongUNetPosEmbd): https://github.com/NVIDIA/physicsnemo/tree/main/physicsnemo/models/diffusion  
"""
Song-style U-Net with global positional embedding grid (SongUNetPosEmbd reimplementation).
Author:  (reimplementation inspired by NVIDIA PhysicsNeMo docs/source)
"""

from typing import List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



# -- Utilities ----------------------------
def exists(x):
    return x is not None

def default(x, d):
    return x if exists(x) else d

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# -- Timestep embedding ----------------------------
def timestep_embedding(t: torch.Tensor, dim: int):
    """t: (B,) tensor of floats (e.g., diffusion step or continuous time). Returns (B, dim)."""
    if t is None:
        return None
    # -- Create sinusoidal features --
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device, dtype=t.dtype) / max(1, half - 1))
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# -- Positional embedding ----------------------------
def build_sinusoidal_pos_emb(H: int, W: int, n_channels: int = 4, device=None, dtype=torch.float32):
    """
    Build a fixed sinusoidal spatial embedding of shape (n_channels, H, W).
    For n_channels=4 we provide [sin(pi*x), cos(pi*x), sin(pi*y), cos(pi*y)] with x,y in [-1,1].
    If n_channels > 4 we add higher-frequency harmonics pairwise.
    """
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    y = torch.linspace(-1.0, 1.0, steps=H, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, steps=W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")  # (H, W)
    emb_list = []

    # generate harmonics: for channel pairs (sin, cos) at frequencies 1,2,4,...
    # we need n_channels to be even; if odd, final channel is zero padded later
    pairs = n_channels // 2
    freqs = [1.0 * (2 ** i) for i in range(pairs)]
    for f in freqs:
        emb_list.append(torch.sin(math.pi * f * xx))
        emb_list.append(torch.cos(math.pi * f * xx))
        if len(emb_list) >= n_channels:
            break
        emb_list.append(torch.sin(math.pi * f * yy))
        emb_list.append(torch.cos(math.pi * f * yy))
        if len(emb_list) >= n_channels:
            break

    emb = torch.stack(emb_list[:n_channels], dim=0)  # (n_channels, H, W)
    if emb.shape[0] < n_channels:
        # pad with zeros
        pad = n_channels - emb.shape[0]
        emb = torch.cat([emb, torch.zeros(pad, H, W, device=device, dtype=dtype)], dim=0)
    return emb

# -- Basic building blocks ----------------------------
class GroupNormAct(nn.Module):
    def __init__(self, dim, groups=8):
        super().__init__()
        self.gn = nn.GroupNorm(groups, dim, eps=1e-6, affine=True)
        self.act = SiLU()

    def forward(self, x):
        return self.act(self.gn(x))

# -------------------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim: Optional[int] = None, groups: int = 8):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.norm1 = GroupNormAct(in_ch, groups=groups)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = GroupNormAct(out_ch, groups=groups)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        if exists(time_emb_dim):
            self.time_proj = nn.Linear(time_emb_dim, out_ch)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb: Optional[torch.Tensor] = None):
        h = self.conv1(self.norm1(x))
        if exists(self.time_emb_dim) and exists(t_emb):
            # Add (broadcasted) time embedding
            h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.norm2(h))
        return self.shortcut(x) + h

# -------------------------------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, ch, num_heads=1):
        super().__init__()
        assert ch % num_heads == 0
        self.num_heads = num_heads
        self.scale = (ch // num_heads) ** -0.5
        # use conv1d-like on flattened spatial axis
        self.to_qkv = nn.Conv1d(ch, ch * 3, kernel_size=1)
        self.to_out = nn.Conv1d(ch, ch, kernel_size=1)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        n = H * W
        x_flat = x.view(B, C, n)
        qkv = self.to_qkv(x_flat)  # (B, 3C, n)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(B, self.num_heads, C // self.num_heads, n)
        k = k.view(B, self.num_heads, C // self.num_heads, n)
        v = v.view(B, self.num_heads, C // self.num_heads, n)
        attn = torch.einsum("bhcn,bhcm->bhnm", q * self.scale, k)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        out = out.reshape(B, C, n)
        out = self.to_out(out)
        out = out.view(B, C, H, W)
        return out

# -------------------------------------------------------------------------------------
class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)

# -------------------------------------------------------------------------------------
class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)



# -- SongUNetPosEmbd (both modes) -------------------------------------------------------------------------------------
class SongUNetPosEmbd(nn.Module):
    """
    Song-style U-Net with sinusoidal spatial positional embeddings.
    - in_channels: number of data input channels (excluding context and pos emb)
    - cond_channels_low / cond_channels_high: optional context channels (low/high res)
    - n_pos_emb: number of positional embedding channels (sinusoidal)
    - base_channels: starting number of channels
    - channel_mult: multipliers for each downsample level
    - num_blocks: residual blocks per level
    - attention_resolutions: list of integer spatial sizes (H or W) where attention is applied
    - time_emb_dim: dimension for timestep embedding (if None or use_time_emb=False, acts as zero)
    - use_time_emb: if True, uses timestep embedding when t is provided; if False, time embedding is zero
    """

    def __init__(
        self,
        *,
        img_resolution: int,
        in_channels: int,
        cond_channels_low: int = 0,
        cond_channels_high: int = 0,
        n_pos_emb: int = 4,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mult: Optional[List[int]] = None,
        num_blocks: int = 2,
        attention_resolutions: Optional[List[int]] = None,
        time_emb_dim: Optional[int] = None,
        use_time_emb: bool = False,
    ):
        super().__init__()

        # -- SELF --
        channel_mult = default(channel_mult, [1, 2, 2, 4])
        attention_resolutions = default(attention_resolutions, [])
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.cond_channels_low = cond_channels_low
        self.cond_channels_high = cond_channels_high
        self.n_pos_emb = n_pos_emb
        self.use_time_emb = use_time_emb and exists(time_emb_dim)
        self.time_emb_dim = time_emb_dim if self.use_time_emb else None

        # -- Total input channels after concatenation (data + contexts + pos emb) --
        total_in = in_channels + cond_channels_low + cond_channels_high + n_pos_emb
        print(f"Total input channels: {total_in}")

        # -- Step (i.e., sigma) embedding MLP (optional, only in denoiser/generator mode) --
        if self.use_time_emb:
            self.time_mlp = nn.Sequential(
                nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                SiLU(),
                nn.Linear(self.time_emb_dim * 4, self.time_emb_dim),
            )
        else:
            self.time_mlp = None

        # -- Initial conv layer --
        self.init_conv = nn.Conv2d(total_in, base_channels, kernel_size=3, padding=1)

        # -- Encoder --
        self.downs = nn.ModuleList()
        self.downsample_ops = nn.ModuleList()
        ch = base_channels
        curr_res = img_resolution
        encoder_res_channels = {}
        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_blocks):
                blocks.append(ResidualBlock(ch, out_ch, time_emb_dim=self.time_emb_dim))
                ch = out_ch
            self.downs.append(blocks)
            encoder_res_channels[curr_res] = ch
            if i != len(channel_mult) - 1:
                self.downsample_ops.append(Downsample(ch, ch))
                curr_res = curr_res // 2

        # -- Middle --
        self.mid1 = ResidualBlock(ch, ch, time_emb_dim=self.time_emb_dim)
        self.mid_attn = SelfAttention(ch, num_heads=1) if (curr_res in attention_resolutions) else None
        self.mid2 = ResidualBlock(ch, ch, time_emb_dim=self.time_emb_dim)

        # -- Decoder --
        self.ups = nn.ModuleList()
        self.upsample_ops = nn.ModuleList()
        decoder_res_channels = {}
        for i, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            # first block gets concatenated skip, so input channels = ch + out_ch
            blocks.append(ResidualBlock(ch + out_ch, out_ch, time_emb_dim=self.time_emb_dim))
            ch = out_ch
            for _ in range(num_blocks - 1):
                blocks.append(ResidualBlock(ch, out_ch, time_emb_dim=self.time_emb_dim))
            self.ups.append(blocks)
            decoder_res_channels[curr_res] = ch
            if i != 0:
                self.upsample_ops.append(Upsample(ch, ch))
                curr_res = curr_res * 2

        # -- Final conv layer --
        self.final_norm = GroupNormAct(ch, groups=8)
        self.final_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

        # -- Attention layers --
        self.attn_res_set = set(attention_resolutions)
        self.encoder_attns = nn.ModuleDict()
        self.decoder_attns = nn.ModuleDict()
        for res in attention_resolutions:
            self.encoder_attns[str(res)] = SelfAttention(encoder_res_channels[res], num_heads=1)
            self.decoder_attns[str(res)] = SelfAttention(decoder_res_channels[res], num_heads=1)


    # -------------------------------------------------------------------------------------
    def _prepare_inputs(
        self,
        x: Optional[torch.Tensor],
        context_low: Optional[torch.Tensor],
        context_high: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        x: (B, in_channels, H, W)
        context_low: (B, cond_channels_low, H_low, W_low) OR None
        context_high: (B, cond_channels_high, H, W) OR None
        Returns concatenated tensor (B, total_in, H, W)
        """

        H = self.img_resolution
        W = self.img_resolution
        parts = []

        # --- Noisy fields ---
        # print(f"Input channels: {self.in_channels}")
        if self.in_channels != 0 and exists(x):
            # --- Check spatial resolution of noisy fields ---
            B, _, H_n, W_n = x.shape
            assert H_n == self.img_resolution and W_n == self.img_resolution, (
                f"Expected resolution {self.img_resolution}x{self.img_resolution}, got {H_n}x{W_n}"
            )
            parts.append(x)
        # elif self.in_channels != 0:
        #     # if expected but not provided, append zeros
        #     parts.append(torch.zeros(B, self.cond_channels_low, H, W, device=x.device, dtype=x.dtype))

        # --- Low-resolution context ---
        # print(f"Cond channels low: {self.cond_channels_low}")
        if self.cond_channels_low != 0 and exists(context_low):
            # upsample low-res context to full resolution
            ctx_low_up = F.interpolate(context_low, size=(H, W), mode="bilinear", align_corners=False)
            parts.append(ctx_low_up)
        # elif self.cond_channels_low != 0:
        #     # if expected but not provided, append zeros
        #     parts.append(torch.zeros(B, self.cond_channels_low, H, W, device=x.device, dtype=x.dtype))

        # --- High-resolution context ---
        # print(f"Cond channels high: {self.cond_channels_high}")
        if self.cond_channels_high != 0 and exists(context_high):
            parts.append(context_high)
        # elif self.cond_channels_high != 0:
        #     # if expected but not provided, append zeros
        #     parts.append(torch.zeros(B, self.cond_channels_high, H, W, device=x.device, dtype=x.dtype))

        # --- Positional embeddings ---
        B = parts[0].shape[0]
        pos_emb = build_sinusoidal_pos_emb(H, W, self.n_pos_emb, device=parts[0].device, dtype=parts[0].dtype)
        pos_emb = pos_emb[None].repeat(B, 1, 1, 1)  # (B, n_pos_emb, H, W)
        parts.append(pos_emb)
        x_cat = torch.cat(parts, dim=1)
        # print(f"x_cat: {x_cat.shape}")
        return x_cat

    # -------------------------------------------------------------------------------------
    def forward(
        self,
        context_low: Optional[torch.Tensor] = None,
        context_high: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None, # i.e., continuous diffusion timestep or sigma step
    ) -> torch.Tensor:
        """
        context_low: optional (B, cond_channels_low, H_low, W_low)
        context_high: optional (B, cond_channels_high, H, W)
        x: optional (B, in_channels, H, W)
        t: optional (B,) continuous diffusion timestep (scaled appropriately).
           If use_time_emb==False or t is None, time embedding behaves as zero.
        Returns: (B, out_channels, H, W)
        """

        # # -- Get shapes --
        # B, _, H, W = x.shape

        # -- Concatenate noise, contexts, and positional embeddings --
        h = self._prepare_inputs(x, context_low, context_high)

        # -- Prepare denoiser step (i.e., sigma) embedding --
        t_emb = None
        if self.use_time_emb and t is not None:
            t_scaled = t.float()
            t_emb = timestep_embedding(t_scaled, self.time_emb_dim)
            t_emb = self.time_mlp(t_emb) if self.time_mlp is not None else t_emb
        elif self.use_time_emb and t is None:
            assert False, "Time embedding is enabled but no timestep t is provided."

        # -- Initial conv layer --
        h = self.init_conv(h)

        # -- Encoder --
        hs = []
        level_res = self.img_resolution
        for i, blocks in enumerate(self.downs):
            for block in blocks:
                h = block(h, t_emb)
            hs.append(h)
            if level_res in self.attn_res_set:
                h = h + self.encoder_attns[str(level_res)](h)
            if i < len(self.downs) - 1:
                h = self.downsample_ops[i](h)
                level_res = level_res // 2

        # -- Middle --
        h = self.mid1(h, t_emb)
        if exists(self.mid_attn):
            h = h + self.mid_attn(h)
        h = self.mid2(h, t_emb)

        # -- Decoder --
        for i, blocks in enumerate(self.ups):
            skip = hs.pop()
            # concat skip and feed first block
            h = torch.cat([h, skip], dim=1)
            h = blocks[0](h, t_emb)
            for block in blocks[1:]:
                h = block(h, t_emb)
            # optional attention at the current resolution
            res = h.shape[-1]
            if h.shape[-1] in self.attn_res_set:
                h = h + self.decoder_attns[str(res)](h)
            if i < len(self.ups) - 1:
                h = self.upsample_ops[i](h)

        # -- Final conv layer --
        out = self.final_conv(self.final_norm(h))

        # -- Return --
        return out


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    B = 2
    H = W = 128
    in_ch = 2
    cond_low = 15
    cond_high = 2
    model = SongUNetPosEmbd(
        img_resolution=H,
        in_channels=in_ch,
        cond_channels_low=cond_low,
        cond_channels_high=cond_high,
        n_pos_emb=4,
        out_channels=2,
        base_channels=64,
        channel_mult=[1, 2, 2, 4],
        num_blocks=2,
        attention_resolutions=[16],
        time_emb_dim=128,
        use_time_emb=True,  # if you want diffusion mode with timestep embedding
    )

    # create dummy inputs
    x = torch.randn(B, in_ch, H, W)
    ctx_low = torch.randn(B, cond_low, H // 4, W // 4)  # example low-res context
    ctx_high = torch.randn(B, cond_high, H, W)

    # deterministic regression mode (no timestep)
    y_reg = model(x, context_low=ctx_low, context_high=ctx_high, t=None)
    print("reg output:", y_reg.shape)

    # diffusion denoising mode (with timestep)
    t = torch.rand(B)  # e.g., continuous scaled timestep
    y_diff = model(x, context_low=ctx_low, context_high=ctx_high, t=t)
    print("diff output:", y_diff.shape)