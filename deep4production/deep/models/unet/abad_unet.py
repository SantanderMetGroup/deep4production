import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from deep4production.utils.general import get_func_from_string


class UnitConv(nn.Module):
    """
    2D convolution block with optional batch normalization and ReLU activations.
    Parameters:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size.
        padding (int): Padding.
        batch_norm (bool): Use batch normalization.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 padding: int, batch_norm: bool):
        super().__init__()
 
        if batch_norm:
            self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU())
        else:

            self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.ReLU())

    def forward(self, x):
        """
        Forward pass for UnitConv block.
        Parameters:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv(x)

class UpLayer(nn.Module):
    """
    Upsampling block using transposed convolution or upsampling + convolution.
    Parameters:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        trans_conv (bool): Use transposed convolution.
    """
    
    def __init__(self, in_channels: int, out_channels: int, trans_conv: bool):
        super().__init__()
 
        if trans_conv:
            self.layer_op = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=2, stride=2)
        else:
            self.layer_op = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )

    def forward(self, x):
        """
        Forward pass for UpLayer block.
        Parameters:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.layer_op(x)


class abad_unet(torch.nn.Module):
    """
    UNet-style model for deep learning downscaling.
    Purpose: Encodes and decodes spatial features for climate prediction.
    Parameters:
        x_shape (tuple): Input shape.
        y_shape (tuple): Output shape.
        input_padding: Padding for input.
        kernel_size (int): Kernel size.
        padding: Padding value.
        batch_norm (bool): Use batch normalization.
        trans_conv (bool): Use transposed convolution.
        num_final_res_increases (int): Number of final upsampling layers.
        base_channels (int): Base channels.
        loss_function_name (str): Loss function name.
        output_activation (dict): Output activation specification.
    """

    def __init__(self, x_shape, y_shape, 
                 input_padding, kernel_size, padding,
                 batch_norm, trans_conv, num_final_res_increases,
                 base_channels: int=64,
                 loss_function_name: str=None,
                 output_activation: dict | None = None):

        super().__init__()

        ## --- Predictor checks ---
        if (len(x_shape) != 3):
            error_msg =\
            'X must have a dimension of length 3: (C, H, W)'
            raise ValueError(error_msg)
        self.num_input_vars, *self.spatial = x_shape

        ### --- Predictand checks ---
        if (len(y_shape) != 3):
            error_msg =\
            'Y must have a dimension of length 3: (C, H, W)'
            raise ValueError(error_msg)
        self.num_output_vars, *self.spatial = y_shape

        ## --- SELF: Model parameters ---
        self.loss_function_name = loss_function_name
        self.input_padding = input_padding
        self.kernel_size = int(kernel_size)
        self.padding = padding
        self.batch_norm = batch_norm
        self.trans_conv = trans_conv
        self.num_final_res_increases = num_final_res_increases
        self.base_channels = base_channels

        # --- Default activation (no change to output) ---
        default_act = nn.Identity()

        ## --- Encoder ---
        self.down_conv_1 = UnitConv(in_channels=self.num_input_vars, out_channels=base_channels, kernel_size=self.kernel_size, padding=self.padding, batch_norm=self.batch_norm)
        self.maxpool_1 = nn.MaxPool2d((2, 2))

        self.down_conv_2 = UnitConv(in_channels=base_channels, out_channels=base_channels*2, kernel_size=self.kernel_size, padding=self.padding, batch_norm=self.batch_norm)
        self.maxpool_2 = nn.MaxPool2d((2, 2))

        self.down_conv_3 = UnitConv(in_channels=base_channels*2, out_channels=base_channels*4, kernel_size=self.kernel_size, padding=self.padding, batch_norm=self.batch_norm)
        self.maxpool_3 = nn.MaxPool2d((2, 2))

        self.down_conv_4 = UnitConv(in_channels=base_channels*4, out_channels=base_channels*8, kernel_size=self.kernel_size, padding=self.padding, batch_norm=self.batch_norm)
        self.maxpool_4 = nn.MaxPool2d((2, 2))

        ## --- Decoder ---
        self.trans_conv_1 = UpLayer(in_channels=base_channels*8, out_channels=base_channels*4, trans_conv=self.trans_conv)
        self.up_conv_1 = UnitConv(in_channels=base_channels*8, out_channels=base_channels*4, kernel_size=self.kernel_size, padding=self.padding, batch_norm=self.batch_norm)

        self.trans_conv_2 = UpLayer(in_channels=base_channels*4, out_channels=base_channels*2, trans_conv=self.trans_conv)
        self.up_conv_2 = UnitConv(in_channels=base_channels*4, out_channels=base_channels*2, kernel_size=self.kernel_size, padding=self.padding, batch_norm=self.batch_norm)

        self.trans_conv_3 = UpLayer(in_channels=base_channels*2, out_channels=base_channels, trans_conv=self.trans_conv)
        self.up_conv_3 = UnitConv(in_channels=base_channels*2, out_channels=base_channels, kernel_size=self.kernel_size, padding=self.padding, batch_norm=self.batch_norm)

        ## --- Final upsampling segment ---
        for i in range(self.num_final_res_increases):
            setattr(self, f"trans_conv_{i+4}", UpLayer(in_channels=base_channels, out_channels=base_channels,
                                                      trans_conv=self.trans_conv))
            setattr(self, f"up_conv_{i+4}", UnitConv(in_channels=base_channels, out_channels=base_channels,
                                                     kernel_size=self.kernel_size, padding=self.padding,
                                                     batch_norm=self.batch_norm))
  
        ## --- Output layers ---
        if self.loss_function_name == "NLLGaussianLoss":
            self.out_mean = nn.Conv2d(in_channels=base_channels, out_channels=self.num_output_vars, kernel_size=1)
            self.out_log_var = nn.Conv2d(in_channels=base_channels, out_channels=self.num_output_vars, kernel_size=1)
        elif self.loss_function_name == "NLLBerGammaLoss": 
            self.p = nn.Conv2d(in_channels=base_channels, out_channels=1, kernel_size=1)
            self.log_shape = nn.Conv2d(in_channels=base_channels, out_channels=1, kernel_size=1)
            self.log_scale = nn.Conv2d(in_channels=base_channels, out_channels=1, kernel_size=1)
        else:
            self.out = nn.Conv2d(in_channels=base_channels, out_channels=self.num_output_vars, kernel_size=1)
        
        # --- Per-variable activations ---
        self.output_activation = nn.ModuleDict()
        self._activation_map = {i: nn.Identity() for i in range(self.num_output_vars)}  # default: linear
        if output_activation:
            for var_name, spec in output_activation.items():
                idx = spec["idx"]
                act_class = get_func_from_string(spec.get("module", "torch.nn"), spec["name"], kwargs = spec.get("kwargs", None))
                act = act_class if isinstance(act_class, torch.nn.Module) else act_class()
                self.output_activation[var_name] = act
                self._activation_map[idx] = act

    def forward(self, x):
        """
        Forward pass for abad_unet model.
        Parameters:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        B = x.shape[0]
        x = F.pad(x, self.input_padding)

        ## --- Encoder ---
        x1 = self.down_conv_1(x)
        x1_maxpool = self.maxpool_1(x1)
        x2 = self.down_conv_2(x1_maxpool)
        x2_maxpool = self.maxpool_2(x2)
        x3 = self.down_conv_3(x2_maxpool)
        x3_maxpool = self.maxpool_3(x3)
        x4 = self.down_conv_4(x3_maxpool)

        ## --- Decoder ---
        x5 = self.trans_conv_1(x4)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.up_conv_1(x5)
        x6 = self.trans_conv_2(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.up_conv_2(x6)
        x7 = self.trans_conv_3(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.up_conv_3(x7)

        ## --- Final upsampling segment ---
        x_out = x7
        for i in range(self.num_final_res_increases):
            trans_conv = getattr(self, f"trans_conv_{i+4}")
            up_conv = getattr(self, f"up_conv_{i+4}")
            x_out = trans_conv(x_out)
            x_out = up_conv(x_out)

        ## --- Output layers ---
        if self.loss_function_name == "NLLGaussianLoss":
            mean = self.out_mean(x_out).view(B, self.num_output_vars, 1, *self.spatial) # (batch_size, channel, 1, *spatial)
            log_var = self.out_log_var(x_out).view(B, self.num_output_vars, 1, *self.spatial) # (batch_size, channel, 1, *spatial)
            out = torch.cat((mean, log_var), dim=2)
        elif self.loss_function_name == "NLLBerGammaLoss":
            p = self.p(x_out).view(B, 1, *self.spatial) # (batch_size, 1, *spatial) 
            p = torch.sigmoid(p)
            log_shape = self.log_shape(x_out).view(B, 1, *self.spatial) # (batch_size, 1, *spatial) 
            log_scale = self.log_scale(x_out).view(B, 1, *self.spatial) # (batch_size, 1, *spatial) 
            out = torch.cat((p, log_shape, log_scale), dim = 1)
        else:
            out_default = self.out(x_out)
            # --- Apply per-channel activations ---
            activated = []
            for idx in range(out_default.shape[1]):
                act = self._activation_map.get(idx, nn.Identity())
                activated.append(act(out_default[:, idx:idx+1, ...]))
            out = torch.cat(activated, dim=1)
        # --- Return ---
        return out