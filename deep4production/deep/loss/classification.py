"""
Binary classification loss functions (BCE, focal loss).

Authors:
    Jose González-Abad
    Alfonso Hernanz
    Jorge Baño-Medina
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BinaryCrossEntropyLoss(nn.Module):
    """
    N-dimensional Binary Cross Entropy loss.
    Purpose: Computes BCE for binary targets, supports NaN masking.
    Parameters:
        threshold (float): Threshold for binarization.
        ignore_nans (bool): Ignore NaNs in target domain.
    """

    def __init__(self, threshold: 1.0, ignore_nans: bool = False):
        super().__init__()
        # Allow single float or list/tuple of floats
        if isinstance(threshold, (float, int)):
            self.threshold = threshold
        elif isinstance(threshold, (list, tuple)):
            self.threshold = torch.tensor(threshold, dtype=torch.float32)
        else:
            raise TypeError(
                f"threshold must be float, list, or tuple, got {type(threshold)}"
            )
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Computes BCE loss between target and output.
        Parameters:
            target (torch.Tensor): Target data.
            output (torch.Tensor): Model output.
        Returns:
            torch.Tensor: Loss value.
        """

        # --- Check shapes match -----------------------------
        if target.shape != output.shape:
            raise ValueError(
                f"Target and output must have the same shape, got "
                f"target={target.shape}, output={output.shape}"
            )

        # --- Reshape spatial or graph dimensions ------------
        # Target/output become (B, C, G)
        if target.ndim == 4:
            # (B, C, H, W) → (B, C, H*W)
            B, C, H, W = target.shape
            target = target.reshape(B, C, -1)
            output = output.reshape(B, C, -1)
        elif target.ndim == 3:
            B, C, G = target.shape
        else:
            raise ValueError(
                f"Unsupported shape {target.shape}. Must be (B,C,G) or (B,C,H,W)."
            )

        # --- Broadcast threshold(s) to (1, C, 1) ------------------------------
        if isinstance(self.threshold, torch.Tensor):
            # Per-channel thresholds
            if self.threshold.numel() != C:
                raise ValueError(
                    f"Threshold list has {self.threshold.numel()} entries, "
                    f"but input has C={C} channels."
                )
            thr = self.threshold.view(1, C, 1).to(target.device)
        else:
            # Single float threshold for all channels
            thr = torch.tensor(self.threshold, dtype=torch.float32,
                               device=target.device).view(1, 1, 1)

        # --- Binarize target using thresholds ---------------------------------
        target_bin = (target >= thr).float()

        # --- Flatten to vectors per channel ------------------
        # Shape: (B*C*G,)
        target_flat = target.reshape(-1).float()
        output_flat = output.reshape(-1)

        # --- Handle NaNs (ignore them) -----------------------------
        if self.ignore_nans:
            mask = ~torch.isnan(target_flat)
            output_flat = output_flat[mask]
            target_flat = target_flat[mask]

        # --- Cross-entropy expects class indices (long) -----------------------------
        target_flat = target_flat.long()

        # --- Bernoulli NLL = Binary cross-entropy w/ logits ---
        loss = F.binary_cross_entropy_with_logits(output_flat, target_flat)

        return loss


class BernoulliFocalLoss(nn.Module):
    """
    N-dimensional Bernoulli Focal Loss.
    Purpose: Computes focal loss for binary targets, supports NaN masking.
    Parameters:
        gamma (float): Focusing parameter.
        alpha (float): Weight for positive class.
        threshold (float): Threshold for binarization.
        ignore_nans (bool): Ignore NaNs in target domain.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, threshold: float = 1.0, ignore_nans: bool = False):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.threshold = threshold
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Computes Bernoulli focal loss between target and output.
        Parameters:
            target (torch.Tensor): Target data.
            output (torch.Tensor): Model output.
        Returns:
            torch.Tensor: Loss value.
        """

        if target.shape != output.shape:
            raise ValueError(f"Target and output must match shapes, got {target.shape} vs {output.shape}")

        # --- Reshape to (B, C, G) ----------------------------------------
        if target.ndim == 4:
            B, C, H, W = target.shape
            target = target.reshape(B, C, -1)
            output = output.reshape(B, C, -1)
        elif target.ndim == 3:
            B, C, G = target.shape
        else:
            raise ValueError(f"Unsupported shape {target.shape}. Must be (B,C,G) or (B,C,H,W)")

        # --- Binarize target using the same threshold for all channels ---
        target_bin = (target >= self.threshold).float()

        # --- Flatten for computation --------------------------------------
        target_flat = target_bin.reshape(-1)
        output_flat = output.reshape(-1)

        # --- NaN masking --------------------------------------------------
        if self.ignore_nans:
            mask = ~torch.isnan(target_flat)
            target_flat = target_flat[mask]
            output_flat = output_flat[mask]

        # --- Compute probabilities ---------------------------------------
        p = torch.sigmoid(output_flat)
        pt = p * target_flat + (1 - p) * (1 - target_flat)

        # --- Compute focal weight and alpha_t ----------------------------------------
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = self.alpha * target_flat + (1 - self.alpha) * (1 - target_flat)

        # --- Compute focal loss ------------------------------------------
        loss = -alpha_t * focal_weight * torch.log(pt + 1e-8)

        return loss.mean()
