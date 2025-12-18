import torch
import torch.nn.init as init
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from deep4downscaling.utils.general import get_func_from_string

def impute_padding(kernel_size, dilation=1):
    return dilation * (kernel_size - 1) // 2

class ConditionalLayerNorm2d(nn.Module):
    """
    Spatially varying conditional LayerNorm.

    Noise is defined per grid cell and mapped to
    per-channel scale (gamma) and shift (beta).
    """
    def __init__(self, num_features, noise_dim):
        super().__init__()

        # LayerNorm over channels (C), no learned affine parameters
        self.ln = nn.LayerNorm(num_features, elementwise_affine=False)

        # Noise injector: 1x1 conv = per-pixel MLP
        self.noise_mlp = nn.Sequential(
            nn.Conv2d(noise_dim, 2 * num_features, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(2 * num_features, 2 * num_features, kernel_size=1)
        )

    def forward(self, x, noise):
        """
        x     : (B, C, H, W)    latent feature map
        noise : (B, Z, H, W)    Gaussian noise per grid cell
        """
        B, C, H, W = x.shape

        # ---- 1. Normalize features across channels ----
        # Move channels to last dimension for LayerNorm
        x = x.permute(0, 2, 3, 1)      # (B, H, W, C)
        x_norm = self.ln(x)            # (B, H, W, C)

        # ---- 2. Map noise -> gamma, beta ----
        gamma_beta = self.noise_mlp(noise)  # (B, 2C, H, W)
        gamma, beta = gamma_beta.chunk(2, dim=1)

        # ---- 3. Match dimensions for modulation ----
        gamma = gamma.permute(0, 2, 3, 1)   # (B, H, W, C)
        beta  = beta.permute(0, 2, 3, 1)    # (B, H, W, C)

        # ---- 4. Apply conditional modulation ----
        out = gamma * x_norm + beta         # (B, H, W, C)

        # ---- 5. Restore channel-first format ----
        return out.permute(0, 3, 1, 2)      # (B, C, H, W)


class Block(nn.Module):
    """
    DeepESD convolutional block with stochastic noise injection
    via spatially varying conditional LayerNorm.

    This block performs:
      1) A spatial convolution to extract features
      2) Optional stochastic modulation of features using Gaussian noise
         injected through conditional LayerNorm (per grid cell)
      3) A non-linear activation (ReLU)

    If sigma == 0, the block reduces to a standard Conv + BatchNorm + ReLU.
    If sigma > 0, BatchNorm is replaced by ConditionalLayerNorm2d and
    Gaussian noise is injected in the latent space.
    """

    def __init__(self, in_channels, out_channels, kernel_size, noise_inject=True, noise_dim=4):
        super().__init__()

        # ---- Convolution layer ----
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=impute_padding(kernel_size)
        )

        # ---- Noise parameters ----
        self.noise_inject = noise_inject
        self.noise_dim = noise_dim

        # ---- Normalization / Noise injection ----
        if noise_inject:
            # Conditional LayerNorm modulated by spatial Gaussian noise
            self.cond_ln = ConditionalLayerNorm2d(
                num_features=out_channels,
                noise_dim=noise_dim
            )
        else:
            # Standard deterministic BatchNorm
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, noise=None):
        """
        x: (B, C_in, H, W)
        """
        # ---- 1. Convolution ----
        x = self.conv(x)  # (B, C_out, H, W)

        # ---- 2. Noise injection or normalization ----
        if self.noise_inject:
            # ---- Inject noise via conditional LayerNorm ----
            x = self.cond_ln(x, noise)
        else:
            # ---- Apply BatchNorm ----
            x = self.bn(x)

        # ---- 3. Non-linearity ----
        x = F.relu(x)

        # ---- 4. Return transformed features ----
        return x

class DeepESDcrps(torch.nn.Module):

    """
    DeepESD model as proposed in Baño-Medina et al. 2024. 

    Baño-Medina, J., Manzanas, R., Cimadevilla, E., Fernández, J., González-Abad,
    J., Cofiño, A. S., and Gutiérrez, J. M.: Downscaling multi-model climate projection
    ensembles with deep learning (DeepESD): contribution to CORDEX EUR-44, Geosci. Model
    Dev., 15, 6747–6758, https://doi.org/10.5194/gmd-15-6747-2022, 2022.

    """

    def __init__(self, 
                 x_shape,
                 y_shape,
                 f_shape: list[int]=None,
                 filters: list[int]=[50,25,10],
                 kernel_size: int=3,
                 sigma: float=0.,
                 loss_function_name: str=None,
                 output_activation: dict = None):

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

        ## --- Noise (for minimizing CRPS) ---
        if sigma > 0:
            noise_inject = True
            self.sigma = sigma
            self.noise_dim = 4
        else:
            noise_inject = False
            self.sigma = 0.0
        self.noise_inject = noise_inject

        ## --- Hidden layers ---
        self.block_1 = Block(num_input_vars, filters[0], kernel_size, noise_inject=noise_inject)
        self.block_2 = Block(filters[0], filters[1], kernel_size, noise_inject=noise_inject)
        self.block_3 = Block(filters[1], filters[2], kernel_size, noise_inject=noise_inject)

        ## --- Forcing ---
        self.f_shape = f_shape
        if f_shape is not None:
            input_forcing_features = int(np.prod(f_shape))
            flatten_features = H * W * filters[-1]
            self.mlp_forcing = torch.nn.Linear(in_features=input_forcing_features, out_features=flatten_features)

        ## --- Output layers ---
        number_neurons_last_hidden = H * W * filters[-1]
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
        B = x.size(0)

        # --- Inject noise? ---
        if self.noise_inject:
            # Extract spatial dimensions
            B, C, H, W = x.shape
            # Sample spatially varying Gaussian noise per grid cell
            noise = torch.randn(
                B, self.noise_dim, H, W, device=x.device
            ) * self.sigma
        else:
            noise = None

        # --- First part: input and hidden layers ---
        x = self.block_1(x, noise)
        x = self.block_2(x, noise)
        x = self.block_3(x, noise)

        # --- Flatten ---
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


