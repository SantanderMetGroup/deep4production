"""
CRPS-based loss functions for ensemble predictions (spatial and spectral).

Authors:
    Jose González-Abad
    Alfonso Hernanz
    Jorge Baño-Medina
"""

import torch
import torch.nn as nn


### -------------------------------------------------------------------------------- ###
### -------------------- CRPS Spectral Loss --------------------------------------- ###
class CRPSSpectralLoss(nn.Module):
    """
    Fair Continuous Ranked Probability Score (CRPS) with spectral component.
    Purpose: Computes CRPS and spectral CRPS for ensemble predictions.
    Parameters:
        ignore_nans (bool): Ignore NaNs in target domain.
        H_shape (int): Height of spatial domain.
        W_shape (int): Width of spatial domain.
        beta (int): Power parameter for CRPS.
        lambda_spectral (float): Weight for spectral CRPS.
        spatial_resolution (float): Spatial resolution for filtering.
    """

    def __init__(
        self,
        ignore_nans: bool,
        H_shape: int,
        W_shape: int,
        beta: int = 1,
        lambda_spectral: float = 0.1,
        spatial_resolution: float = None,
    ) -> None:
        super(CRPSSpectralLoss, self).__init__()
        self.ignore_nans = ignore_nans
        self.H_shape = H_shape
        self.W_shape = W_shape
        self.beta = beta
        self.lambda_spectral = lambda_spectral
        if spatial_resolution is not None and spatial_resolution <= 0:
            raise ValueError("spatial_resolution must be > 0 when provided.")
        self.spatial_resolution = spatial_resolution
        self.filter_nans = (
            False  # Control whether to filter out nans in _CRPS_pointwise
        )

    def _CRPS_pointwise(self, target: torch.Tensor, output) -> torch.Tensor:
        """
        Computes pointwise CRPS for ensemble predictions.
        Parameters:
            target (torch.Tensor): Target data.
            output (list): List of ensemble predictions.
        Returns:
            torch.Tensor: CRPS value.
        """

        if self.ignore_nans and self.filter_nans:
            nans_idx = torch.isnan(target)
            target = target[~nans_idx]
            output = [out[~nans_idx] for out in output]

        # Number of ensemble members
        M = len(output)

        # Error between target and each prediction
        first_term = 0.0
        for i in range(M):
            first_term += torch.abs(target - output[i]) ** self.beta
        first_term = first_term / M

        # Difference between all pairs of predictions
        if M > 1:
            second_term = 0.0
            for i in range(M):
                for j in range(M):
                    second_term += torch.abs(output[i] - output[j]) ** self.beta
            second_term = second_term / (2 * M * (M - 1))  # Fair CRPS
        else:
            second_term = 0.0

        # Final loss
        loss = torch.mean(first_term - second_term)

        return loss

    def _FFT(self, data: torch.Tensor) -> torch.Tensor:
        """
        Computes FFT for input data, applies low-pass filtering if needed.
        Parameters:
            data (torch.Tensor): Input data.
        Returns:
            torch.Tensor: FFT-transformed data.
        """

        # It does not make sense to filter out nans in the spectral domain
        self.filter_nans = False

        # Fill nans with 0 for the FFT computation
        if isinstance(data, torch.Tensor):  # For the target
            data = [torch.nan_to_num(data, nan=0.0)]
        else:
            data = [torch.nan_to_num(d, nan=0.0) for d in data]

        B = data[0].shape[0]  # Batch size
        if data[0].ndim == 3:
            M = data[0].shape[1]  # Number of ensemble members

        # Reshape to spatial dimensions
        if data[0].ndim == 2:
            data = [member.view(B, self.H_shape, self.W_shape) for member in data]
        elif data[0].ndim == 3:
            data = [member.view(B, M, self.H_shape, self.W_shape) for member in data]

        # Compute FFT
        data = [torch.fft.rfft2(member) for member in data]

        # Optionally remove frequencies beyond the Nyquist limit
        if self.spatial_resolution is not None:
            k_nyquist = 2.0 * torch.pi / (2.0 * self.spatial_resolution)
            kx = (
                2.0
                * torch.pi
                * torch.fft.rfftfreq(self.W_shape, d=self.spatial_resolution)
            )
            ky = (
                2.0
                * torch.pi
                * torch.fft.fftfreq(self.H_shape, d=self.spatial_resolution)
            )
            k_radius = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
            low_pass_mask = k_radius <= k_nyquist
            low_pass_mask = low_pass_mask.to(device=data[0].device)
            data = [member * low_pass_mask for member in data]

        return data

    def forward(self, target: torch.Tensor, output) -> torch.Tensor:
        """
        Computes combined CRPS and spectral CRPS loss.
        Parameters:
            target (torch.Tensor): Target data.
            output (list or torch.Tensor): Ensemble predictions.
        Returns:
            torch.Tensor: Loss value.
        """

        if isinstance(output, torch.Tensor):
            output = [output]

        # Compute standard CRPS
        self.filter_nans = True
        crps_field = self._CRPS_pointwise(target, output)

        # Compute spectral CRPS
        target_fft = self._FFT(target)[0]
        output_fft = self._FFT(output)
        crps_spectral = self._CRPS_pointwise(target_fft, output_fft)

        # Compute total loss
        loss = crps_field + self.lambda_spectral * crps_spectral
        return loss
