import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

########################################################################
# ---- Sinusoidal time embedding ----
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        # Make sure t is [B, 1]
        if t.dim() == 0:
            t = t.unsqueeze(0)         # scalar → [1]
        if t.dim() == 1:
            t = t[:, None]             # [B] → [B, 1]
        elif t.dim() > 2:
            t = t.view(t.shape[0], -1) # flatten extras → [B, N]
            t = t[:, :1]               # keep only 1 value per sample

        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, device=t.device).float() * 
            (torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )

        # Broadcast correctly: [B,1] * [1,half_dim] → [B,half_dim]
        emb = torch.cat(
            [torch.sin(t * freqs[None, :]), torch.cos(t * freqs[None, :])],
            dim=-1
        )
        return self.linear(emb)

# ---- Residual block with FiLM modulation ----
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, embed_dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(embed_dim, out_ch)
        self.cond_proj = nn.Linear(embed_dim, out_ch)

        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, c_emb):
        h = self.conv1(self.act1(self.norm1(x)))

        # FiLM modulation from time + condition embeddings
        film = self.time_proj(t_emb) + self.cond_proj(c_emb)
        h = h + film[:, :, None, None]

        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)

# ---- Small condition encoder ----
class ConditionEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),  # global spatial pooling
        )
        self.proj = nn.Linear(64, embed_dim)

    def forward(self, cond):
        B = cond.shape[0]
        h = self.encoder(cond).view(B, -1)
        return self.proj(h)

# ---- The full lightweight Karras-style UNet ----
class UNetDenoiser(nn.Module):
    def __init__(self, in_channels=1, cond_channels_low=2, cond_channels_high=2, base_channels=64, embed_dim=128, drop_cond_prob=0.1):
        super().__init__()
        # Embeddings
        self.time_embed = TimeEmbedding(embed_dim)
        cond_channels = cond_channels_low + cond_channels_high
        self.cond_enc = ConditionEncoder(cond_channels, embed_dim)

        # Learned null embedding for classifier-free guidance
        self.drop_cond_prob=drop_cond_prob
        self.null_cond = nn.Parameter(torch.zeros(embed_dim))

        # Down
        self.down1 = ResBlock(in_channels, base_channels, embed_dim)
        self.down2 = ResBlock(base_channels, base_channels * 2, embed_dim)
        self.down3 = ResBlock(base_channels * 2, base_channels * 4, embed_dim)
        self.pool = nn.AvgPool2d(2)

        # Bottleneck
        self.mid = ResBlock(base_channels * 4, base_channels * 4, embed_dim)

        # Up
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.res1 = ResBlock(base_channels * 4, base_channels * 2, embed_dim)

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.res2 = ResBlock(base_channels * 2, base_channels, embed_dim)

        self.out = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, x_t, cond_low, cond_high, c_noise, force_uncond=False):
        B = x_t.size(0)

        # Compute embeddings
        # print(sigma_t.shape)
        t_emb = self.time_embed(c_noise.squeeze())
        # print(f"t_emb shape: {t_emb.shape}")    

        # Interpolate low-res condition to match cond_high spatial size
        cond_low_interp = F.interpolate(cond_low, size=cond_high.shape[2:], mode='bilinear', align_corners=False)
        cond = torch.cat([cond_low_interp, cond_high], dim=1)  # concat along channels
        c_emb = self.cond_enc(cond)
        # print(f"c_emb shape: {t_emb.shape}")    

        # --- CFG: unconditional inference mode ---
        if force_uncond:
            c_emb = self.null_cond[None, :].expand(B, -1)

        # --- Training CFG dropout ---
        elif self.training and self.drop_cond_prob > 0:
            mask = (torch.rand(B, 1, device=x_t.device) < self.drop_cond_prob).float()
            c_emb = (1 - mask) * c_emb + mask * self.null_cond[None, :]

        # Down path
        x1 = self.down1(x_t, t_emb, c_emb)
        # print(f"x1 shape: {x1.shape}")   
        x2 = self.down2(self.pool(x1), t_emb, c_emb)
        # print(f"x2 shape: {x2.shape}")   
        x3 = self.down3(self.pool(x2), t_emb, c_emb)
        # print(f"x3 shape: {x3.shape}")   

        x_mid = self.mid(x3, t_emb, c_emb)
        # print(f"x_mid shape: {x_mid.shape}")   

        # Up path
        x = self.up1(x_mid)
        x = self.res1(torch.cat([x, x2], dim=1), t_emb, c_emb)

        x = self.up2(x)
        x = self.res2(torch.cat([x, x1], dim=1), t_emb, c_emb)

        return self.out(x)