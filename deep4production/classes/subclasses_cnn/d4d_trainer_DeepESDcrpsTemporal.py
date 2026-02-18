## Load libraries
import os
import numpy as np
import torch
from torch_geometric.data import HeteroData
## Deep4production
from deep4production.classes.d4d_trainer import d4d_trainer
##################################################################################################################################
class d4d_trainer_custom(d4d_trainer):
    def __init__(self, data, dataloader, id_dir, model_info, graph, d4dpy, Mlflow):
        """
        XX
        """
        ######### Call parent constructor to initialize common attributes #########
        super().__init__(
            data=data,
            dataloader=dataloader,
            id_dir=id_dir,
            model_info=model_info,
            graph=graph,
            d4dpy=d4dpy,
            Mlflow=Mlflow
        )

        self.members = 2    # Ensemble size
        self.noise_states = [None for _ in range(members)]
        self.rho = 0.8   # Temporal correlation coefficient
        self.sigma = 1.0 # Noise scale
        self.noise_dim = 4  # Number of noise channels
        self.noise = None  # Placeholder for noise tensor
        # !!! Note: should rho and sigma be learnable parameters? !!!
        # Initialize one noise state per ensemble member
        

    # -------------------------------------------------------------------------
    def sample_noise(self, x, reset=False):
        """
        Sample temporally correlated (AR(1)) spatial noise.

        Args:
            x: input tensor of shape (B, C, H, W)
            reset: if True, re-initialize noise (e.g. at start of a new sequence)

        Returns:
            noise tensor of shape (B, noise_dim, H, W)
        """
        B, C, H, W = x.shape
        noise_dim = self.noise_dim  # e.g. 4
        sigma = self.sigma          # noise scale
        rho = self.rho              # temporal correlation coefficient (0 < rho < 1)

        # ------------------------------------------------------------------
        # 1. Sample fresh Gaussian innovation
        eps = torch.randn(
            B, noise_dim, H, W, device=x.device
        ) * sigma

        # ------------------------------------------------------------------
        # 2. Initialize or reset noise (start of sequence)
        if reset or self.noise is None:
            noise = eps
        else:
            # ------------------------------------------------------------------
            # 3. AR(1) update: z_t = rho * z_{t-1} + sqrt(1 - rho^2) * eps_t
            noise = rho * self.noise + math.sqrt(1.0 - rho ** 2) * eps

        # ------------------------------------------------------------------
        # 4. Store noise for next timestep
        self.noise = noise.detach()  # detach to avoid backprop through time

        return noise

    # -------------------------------------------------------------------------
    def model_backprop(self, model, data, optimizer, loss_function, device, is_this_training=True):
        """
        Perform a forward + backward pass for one batch with ensemble members.

        Args:
            model: PyTorch model
            data: tuple (x, y, f) from dataset
            optimizer: optimizer
            loss_function: CRPSSpectralLoss or other loss
            device: torch.device
            is_this_training: bool, whether to call backward()
            members: int, ensemble size
        Returns:
            loss value (float)
        """

        # --- Get arrays ---
        x, y, f = data
        x = x.to(device)
        y = y.to(device)

        # --- Forcing ---
        if f[0] != "N/A":
            f = f.to(device)
        else:
            B, Cy, *spatial = y.shape
            f = torch.zeros(1, Cy, *spatial, device=device)

        # --- Loop over samples since we need to sample noise per sample ---
        
        num_samples = x.shape[0]
        prediction_list = []
        for s in range(num_samples):
            # --- Noise ---
            xs = x[s:s+1]  # shape: (1, C, H, W) or (1, C, G)

            # --- Forward pass for each ensemble member ---
            prediction_members = []
            for m in range(self.members):
                # Sample AR(1) noise for this member
                if self.noise_states[m] is None:
                    noise = self.sample_noise(xs, reset=True)
                else:
                    self.noise = self.noise_states[m]
                    noise = self.sample_noise(xs)
                pred_m = model(xs, f, noise)  # shape: (1, C, H, W) or (1, C, G)
                prediction_members.append(pred_m)
                self.noise_states[m] = noise

            # Stack along new ensemble dimension -> (1, M, C, H, W) or (1, M, C, G)
            prediction_ensemble = torch.stack(prediction_members, dim=1)
            # print(f"prediction_ensemble shape: {prediction_ensemble.shape}")
            prediction_list.append(prediction_ensemble)

        # --- Concatenate samples ---
        prediction = torch.cat(prediction_list, dim=0)  # (B, M, C, H, W) or (B, M, C, G)
        # print(f"prediction: {prediction.shape}")

        # --- Compute loss ---
        optimizer.zero_grad()
        loss = loss_function(target=y, output=prediction)

        # --- Backpropagation ---
        if is_this_training:
            loss.backward()
            optimizer.step()

        return loss.item()
