# Ref: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025JH000630
import torch
import torch.nn as nn
import torch.nn.functional as F
from deep4production.utils.general import get_func_from_string
from typing import Tuple, List

def impute_padding(kernel_size, dilation=1):
    return dilation * (kernel_size - 1) // 2

class ConvWithLearnableMapsBlock(nn.Module):
    def __init__(self, in_channels, out_channels, img_shape, kernel_size=3, dilation=1, num_pre_maps=10):
        super().__init__()
        padding = impute_padding(kernel_size, dilation)

        # Main convolution
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=padding, dilation=dilation)

        # Learnable maps
        self.pre_maps = nn.Parameter(torch.zeros(1, num_pre_maps, img_shape[0], img_shape[1]))
        self.pre_conv = nn.Conv2d(num_pre_maps, out_channels, kernel_size=1)

        # Batch norm and ReLU activation
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = x + self.pre_conv(self.pre_maps)
        x = self.act(x)
        x = self.bn(x)
        return x


class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, img_shape, kernel_size=3, upscale_factor=2, num_pre_maps=10, use_post_map=False):
        super().__init__()
        padding = impute_padding(kernel_size, dilation=1)

        # Main convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
        # Learnable maps
        self.use_post_map = use_post_map
        if self.use_post_map:
            # One post-map per output channel
            self.post_maps = nn.Parameter(torch.zeros(1, out_channels // (upscale_factor ** 2), img_shape[0] * upscale_factor, img_shape[1] * upscale_factor))
        else:
            # num_pre_maps pre-maps 
            self.pre_maps = nn.Parameter(torch.zeros(1, num_pre_maps, img_shape[0] * upscale_factor, img_shape[1] * upscale_factor))
            self.pre_conv = nn.Conv2d(num_pre_maps, out_channels // (upscale_factor ** 2), kernel_size=1)

        # Batch norm and ReLU activation
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.bn_2 = nn.BatchNorm2d(out_channels // (upscale_factor ** 2))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn_1(x)
        x = self.pixel_shuffle(x)
        if self.use_post_map:
            x = x + self.post_maps
        else:
            x = x + self.pre_conv(self.pre_maps)
        x = self.act(x)
        x = self.bn_2(x)
        return x


class SMHICNN(nn.Module):
    """
    CNN implementing the architecture described in: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025JH000630.

    Key defaults (matching manuscript):
      - 8 convolutional layers, width=50, kernel_size=3
      - Four dilated convolutions with dilation rates [2,4,8,16]
      - Pre-maps for all conv layers except the last, 10 pre-maps each (learnable)
      - Last conv layer uses post-maps: 1 post-map per feature map
      - PixelShuffle upscaling: 3 sequential pixel-shuffle blocks (default upscale factor=2 each)
      - Padding preserved for all convs (same spatial size until pixel-shuffle)
      - conv -> ReLU -> BatchNorm (BatchNorm after activation per manuscript)
      - Bilinear regridding after final pixel shuffle and before output activations
    """

    def __init__(
        self,
        x_shape: Tuple[int, int, int],
        y_shape: Tuple[int, int, int],
        *,
        base_channels: int = 50, 
        kernel_size: int = 3,
        dilation_rates: List[int] = [1, 2, 4, 8, 16], 
        dilated_layer_indices: List[int] = [1, 2, 3, 4, 5],
        pixel_shuffle_blocks: int = 2,
        pixel_shuffle_upscale_factor: int = 2,
        num_pre_maps: int = 10,
        use_post_map_on_last: bool = True,
        loss_function_name: str = None,
        output_activation: dict | None = None
    ):
        """
        x_shape: (C_in, H_in, W_in)  -- Input shape
        y_shape: (C_out, H_out, W_out) -- Output shape. If loss is negative log-likelihood of Bernoulli-Gamma then C_out should be 1 (e.g., precip)
        """
        super().__init__()

        # ---- Some checks ----
        assert len(dilated_layer_indices) == len(dilation_rates)

        # --- Predictor checks ---
        if (len(x_shape) != 3):
            error_msg =\
            'X must have a dimension of length 3: (C, H, W)'
            raise ValueError(error_msg)
        C_in, H_in, W_in = x_shape

        # --- Predictand checks ---
        if (len(y_shape) < 2):
            error_msg =\
            'Y must have a dimension of length 3: (C, H, W) or length 2: (C, GP)'
            raise ValueError(error_msg)
        C_out, *spatial_dims_out = y_shape
        self.C_out = C_out
        self.spatial_dims_out = spatial_dims_out
        
        # --- C_out parameters ---
        self.loss_function_name = loss_function_name
        C_out_params = C_out
        if loss_function_name == "NLLGaussianLoss":
            C_out_params = C_out * 2 # Mean and std
        elif loss_function_name == "NLLBerGammaLoss":
            C_out_params = 3 # p, shape, scale

        # --- Initialize conv blocks + learnable maps (with dilation) ---
        self.conv_blocks = nn.ModuleList()
        in_ch = C_in
        for i in dilated_layer_indices:
            # Map layer index to a dilation rate in given order
            dilation_idx = dilated_layer_indices.index(i)
            dilation = dilation_rates[dilation_idx]
            # Convolutional layer with dilation
            conv = ConvWithLearnableMapsBlock(in_channels=in_ch, 
                             out_channels=base_channels,
                             img_shape=(H_in, W_in),
                             kernel_size=kernel_size, 
                             dilation=dilation,
                             num_pre_maps=num_pre_maps)
            self.conv_blocks.append(conv)
            in_ch = base_channels

        # ---- Pixel shuffle upscaling blocks ----
        self.ps_blocks = nn.ModuleList()
        H_ps = H_in
        W_ps = W_in
        use_post_map = False
        npm = num_pre_maps
        out_ch = base_channels * (pixel_shuffle_upscale_factor ** 2)
        for i in range(pixel_shuffle_blocks):
            img_shape = (H_ps, W_ps)
            if (i+1) == pixel_shuffle_blocks:
                use_post_map = use_post_map_on_last
                npm = C_out_params * (pixel_shuffle_upscale_factor ** 2)
                out_ch = C_out_params * (pixel_shuffle_upscale_factor ** 2)
            ps_block = PixelShuffleBlock(in_channels=base_channels, 
                                        out_channels=out_ch,
                                        img_shape=img_shape,
                                        kernel_size=kernel_size, 
                                        upscale_factor=pixel_shuffle_upscale_factor,
                                        num_pre_maps=npm, 
                                        use_post_map=use_post_map_on_last)
            self.ps_blocks.append(ps_block)
            # Update shape for next block
            H_ps *= pixel_shuffle_upscale_factor
            W_ps *= pixel_shuffle_upscale_factor
            img_shape = (H_ps, W_ps)
        
        # --- Per-variable activations ---
        self.output_activation = nn.ModuleDict()
        self._activation_map = {i: nn.Identity() for i in range(self.C_out)}  # default: linear
        if output_activation:
            for var_name, spec in output_activation.items():
                idx = spec["idx"]
                act_class = get_func_from_string(spec.get("module", "torch.nn"), spec["name"], kwargs = spec.get("kwargs", None))
                act = act_class if isinstance(act_class, torch.nn.Module) else act_class()
                self.output_activation[var_name] = act
                self._activation_map[idx] = act

    def forward(self, x):
        # Number of samples
        B = x.shape[0]

        # Pass through conv blocks
        for block in self.conv_blocks:
            x = block(x)

        # Pass through pixel shuffle blocks
        for ps_block in self.ps_blocks:
            x = ps_block(x)

        # Bilinear interpolation to target grid 
        x = F.interpolate(x, size=tuple(self.spatial_dims_out), mode='bilinear', align_corners=False)

        # Output layers
        if self.loss_function_name == "NLLGaussianLoss":
            mean = x[:,:self.C_out,...].view(B, self.C_out, 1, *self.spatial_dims_out) # (batch_size, channel, 1, *spatial)
            log_var = x[:,self.C_out:,...].view(B, self.C_out, 1, *self.spatial_dims_out) # (batch_size, channel, 1, *spatial)
            out = torch.cat((mean, log_var), dim=2)
        elif self.loss_function_name == "NLLBerGammaLoss":
            p = x[:,0,...].view(B, 1, *self.spatial_dims_out) # (batch_size, 1, *spatial) 
            p = torch.sigmoid(p)
            log_shape = x[:,1,...].view(B, 1, *self.spatial_dims_out) # (batch_size, 1, *spatial) 
            log_scale = x[:,2,...].view(B, 1, *self.spatial_dims_out) # (batch_size, 1, *spatial) 
            out = torch.cat((p, log_shape, log_scale), dim = 1)
        else:
            # Apply per-channel activations
            activated = []
            for idx in range(x.shape[1]):
                act = self._activation_map.get(idx, nn.Identity())
                activated.append(act(x[:, idx:idx+1, ...]))
            out = torch.cat(activated, dim=1)
        # Return 
        return out