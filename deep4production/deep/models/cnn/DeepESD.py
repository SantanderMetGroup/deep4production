import torch
import torch.nn.init as init
import torch.nn as nn
import numpy as np
import math
from deep4production.utils.general import get_func_from_string

def impute_padding(kernel_size, dilation=1):
    """
    Computes padding for convolution given kernel size and dilation.
    Parameters:
        kernel_size (int): Kernel size.
        dilation (int): Dilation rate.
    Returns:
        int: Padding value.
    """
    return dilation * (kernel_size - 1) // 2

class DeepESD(torch.nn.Module):
    """
    DeepESD model for deep learning climate downscaling.
    Purpose: Implements convolutional layers for spatial prediction.
    Parameters:
        x_shape (tuple): Input shape.
        y_shape (tuple): Output shape.
        f_shape (list[int]): Forcing shape.
        filters (list[int]): Filter sizes.
        kernel_size (int): Kernel size.
        loss_function_name (str): Loss function name.
        output_activation (dict): Output activation specification.
    """

    def __init__(self, 
                 x_shape,
                 y_shape,
                 f_shape: list[int]=None,
                 filters: list[int]=[50,25,10],
                 kernel_size: int=3,
                 loss_function_name: str=None,
                 output_activation: dict = None):
        """
        Initializes DeepESD model.
        Parameters:
            x_shape (tuple): Input shape.
            y_shape (tuple): Output shape.
            f_shape (list[int]): Forcing shape.
            filters (list[int]): Filter sizes.
            kernel_size (int): Kernel size.
            loss_function_name (str): Loss function name.
            output_activation (dict): Output activation specification.
        """
        super().__init__()

        ## --- Predictor checks ---
        if (len(x_shape) != 3):
            error_msg =\
            'X must have a dimension of length 3: (C, H, W)'
            raise ValueError(error_msg)
        num_input_vars, H, W = x_shape

        ## --- Predictand checks ---
        if (len(y_shape) < 2):
            error_msg =\
            'Y must have a dimension of length 3: (C, H, W) or length 2: (C, GP)'
            raise ValueError(error_msg)
        self.num_output_vars, *self.spatial = y_shape
        
        ## --- SELF: Model parameters ---
        self.loss_function_name = loss_function_name

        ## --- Hidden layers ---
        self.conv_1 = torch.nn.Conv2d(in_channels=num_input_vars,
                                      out_channels=filters[0],
                                      kernel_size=kernel_size,
                                      padding=impute_padding(kernel_size, dilation=1))

        self.conv_2 = torch.nn.Conv2d(in_channels=filters[0],
                                      out_channels=filters[1],
                                      kernel_size=kernel_size,
                                      padding=impute_padding(kernel_size, dilation=1))

        self.conv_3 = torch.nn.Conv2d(in_channels=filters[1],
                                      out_channels=filters[2],
                                      kernel_size=kernel_size,
                                      padding=impute_padding(kernel_size, dilation=1))

        ## --- Forcing ---
        self.f_shape = f_shape
        if f_shape is not None:
            input_forcing_features = int(np.prod(f_shape))
            flatten_features = H * W * filters[-1]
            self.mlp_forcing = torch.nn.Linear(in_features=input_forcing_features, out_features=flatten_features)

        
        ## --- Output layers ---
        number_neurons_last_hidden = filters[2] * H * W
        number_neurons_output = self.num_output_vars * math.prod(self.spatial)
        if self.loss_function_name == "NLLGaussianLoss":
            self.out_mean = torch.nn.Linear(in_features=number_neurons_last_hidden, out_features=number_neurons_output)
            self.out_log_var = torch.nn.Linear(in_features=number_neurons_last_hidden, out_features=number_neurons_output)
        elif self.loss_function_name == "NLLBerGammaLoss": 
            self.p = torch.nn.Linear(in_features=number_neurons_last_hidden, out_features=number_neurons_output)
            self.log_shape = torch.nn.Linear(in_features=number_neurons_last_hidden, out_features=number_neurons_output)
            self.log_scale = torch.nn.Linear(in_features=number_neurons_last_hidden, out_features=number_neurons_output)
        else:
            self.out = torch.nn.Linear(in_features=number_neurons_last_hidden, out_features=number_neurons_output)

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

    def forward(self, x: torch.Tensor, f: None) -> torch.Tensor:
        """
        Forward pass of the DeepESD model.
        Parameters:
            x (torch.Tensor): Input tensor.
            f (torch.Tensor or None): Forcing tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        B = x.size(0)
        # --- First part: input and hidden layers ---
        x = self.conv_1(x)
        x = torch.relu(x)
        x = self.conv_2(x)
        x = torch.relu(x)
        x = self.conv_3(x)
        x = torch.relu(x)
        x = torch.flatten(x, start_dim=1)
        
        # --- Add forcing ---
        if (f is not None) and (self.f_shape is not None):
            f = torch.flatten(f, start_dim=1)
            x = x + self.mlp_forcing(f)

        # --- Second part: output layer ---
        if self.loss_function_name == "NLLGaussianLoss":
            mean = self.out_mean(x).view(B, self.num_output_vars, 1, *self.spatial) # (batch_size, channel, 1, *spatial)
            log_var = self.out_log_var(x).view(B, self.num_output_vars, 1, *self.spatial) # (batch_size, channel, 1, *spatial)
            out = torch.cat((mean, log_var), dim=2) # (batch_size, channel, parameters=[mean, log_var], *spatial)
        elif self.loss_function_name == "NLLBerGammaLoss":
            p = self.p(x).view(B, 1, *self.spatial) # (batch_size, num_output_vars, *spatial) 
            p = torch.sigmoid(p)
            log_shape = self.log_shape(x).view(B, 1, *self.spatial) # (batch_size, num_output_vars, *spatial) 
            log_scale = self.log_scale(x).view(B, 1, *self.spatial) # (batch_size, num_output_vars, *spatial) 
            out = torch.cat((p, log_shape, log_scale), dim = 1)
        else:
            x = self.out(x) # (batch_size, num_output_vars * num_gridpoints)
            out_default = x.view(B, self.num_output_vars, *self.spatial) # (batch_size, num_output_vars, *spatial) 
            # --- Apply per-channel activations ---
            activated = []
            for idx in range(out_default.shape[1]):
                act = self._activation_map.get(idx, nn.Identity())
                activated.append(act(out_default[:, idx:idx+1, ...]))
            out = torch.cat(activated, dim=1)
            
        # --- Return ---
        return out