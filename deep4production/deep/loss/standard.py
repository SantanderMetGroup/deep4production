"""
Standard regression loss functions (MAE, MSE, Quantised-MSE).

Authors:
    Jose González-Abad
    Alfonso Hernanz
    Jorge Baño-Medina
"""

import torch
import torch.nn as nn
import numpy as np
from deep4production.utils.zarr import open_zarr_store
from deep4production.utils.log import get_logger

log = get_logger("loss.standard")


### ---------------------------------------------------------------------------------------- ###
### -------------------- Mean Absolute Error Loss ------------------------------------------ ###
class MaeLoss(nn.Module):
    """
    Standard Mean Absolute Error (MAE). It is possible to compute
    this metric over a target dataset with nans.

    Parameters
    ----------
    ignore_nans : bool
        Whether to allow the loss function to ignore nans in the
        target domain.

    target : torch.Tensor
        Target/ground-truth data

    output : torch.Tensor
        Predicted data (model's output)
    """

    def __init__(self, ignore_nans: bool) -> None:
        super(MaeLoss, self).__init__()
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Computes MAE loss between target and output.
        Parameters:
            target (torch.Tensor): Target data.
            output (torch.Tensor): Model output.
        Returns:
            torch.Tensor: Loss value.
        """

        # --- Handle both spatial (H, W) and flattened (GP) shapes ---
        if target.ndim > 3:  # stack spatial dimensions
            B, C, H, W = target.shape
            target = target.reshape(B, C, -1)  # From shape: (B, C, H, W) to (B, C, H*W)
        if output.ndim > 3:  # stack spatial dimensions
            B, C, H, W = output.shape
            output = output.reshape(B, C, -1)  # From shape: (B, C, H, W) to (B, C, H*W)

        # --- Remove Nans if present ---
        if self.ignore_nans:
            nans_idx = torch.isnan(target)
            output = output[~nans_idx]
            target = target[~nans_idx]

        # --- Compute loss and return ---
        loss = torch.mean(torch.abs(target - output))
        return loss


### ---------------------------------------------------------------------------------------- ###
### -------------------- Mean Squared Error Loss ------------------------------------------- ###
class MseLoss(nn.Module):
    """
    Standard Mean Square Error (MSE) loss.
    Purpose: Computes MSE between target and output, optionally ignoring NaNs.
    Parameters:
        ignore_nans (bool): Ignore NaNs in target domain.
    """

    def __init__(self, ignore_nans: bool) -> None:
        super(MseLoss, self).__init__()
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Computes MSE loss between target and output.
        Parameters:
            target (torch.Tensor): Target data.
            output (torch.Tensor): Model output.
        Returns:
            torch.Tensor: Loss value.
        """

        # --- Handle both spatial (H, W) and flattened (GP) shapes ---
        if target.ndim > 3:  # stack spatial dimensions
            B, C, H, W = target.shape
            target = target.reshape(B, C, -1)  # From shape: (B, C, H, W) to (B, C, H*W)
        if output.ndim > 3:  # stack spatial dimensions
            B, C, H, W = output.shape
            output = output.reshape(B, C, -1)  # From shape: (B, C, H, W) to (B, C, H*W)

        # --- Remove Nans if present ---
        if self.ignore_nans:
            nans_idx = torch.isnan(target)
            output = output[~nans_idx]
            target = target[~nans_idx]

        # --- Compute loss and return ---
        loss = torch.mean((target - output) ** 2)
        return loss


### -------------------------------------------------------------------------------- ###
### -------------------- Quantised MSE Loss ---------------------------------------- ###
class QuantisedMSELoss(nn.Module):
    """
    Quantised MSE (QMSE) loss plus standard MSE.
    Purpose: Computes QMSE and MSE for quantile bins.
    Parameters:
        zarr_path (str): Path to Zarr store.
        var (str): Target variable.
        alpha (float): Weight for QMSE term.
        n_quantiles (int): Number of quantile bins.
        threshold (float): Threshold for NaN conversion.
        ignore_nans (bool): Ignore NaNs in computations.
    """

    def __init__(
        self,
        zarr_path: str,
        var: str,
        alpha: float = 1.0,
        n_quantiles: int = 10,
        threshold: float = None,
        ignore_nans: bool = True,
    ):
        super().__init__()

        self.alpha = alpha
        self.n_quantiles = n_quantiles
        self.threshold = threshold
        self.ignore_nans = ignore_nans

        # Store inner MSE object
        self.mse = MseLoss(ignore_nans=ignore_nans)

        # -------- Load reference data from Zarr ----------
        z = open_zarr_store(zarr_path, fmt="auto")
        var_idx = z.attrs["variables"][var]
        data = np.array(z["data"][:, var_idx, :]).flatten()  # From (B, C, GP) to (B*GP)
        data = data[~np.isnan(data)]  # remove NaNs if present

        # -------- Compute bin edges from quantiles --------
        self.bin_edges = torch.tensor(
            np.quantile(data, np.linspace(0, 1, n_quantiles + 1)), dtype=torch.float32
        )

    # ------------------------------------------------------------
    # Compute QMSE for a batch
    # ------------------------------------------------------------
    def _compute_qmse(self, target: torch.Tensor, output: torch.Tensor):
        """
        Computes QMSE for a batch.
        Parameters:
            target (torch.Tensor): Target data.
            output (torch.Tensor): Model output.
        Returns:
            torch.Tensor: QMSE value.
        """

        # ----- reshape (B,C,H,W) → (B*C*G) -----
        if target.ndim > 3:
            B, C, H, W = target.shape
            target = target.reshape(B, C, -1)
            output = output.reshape(B, C, -1)

        # Flatten completely
        target = target.reshape(-1)
        output = output.reshape(-1)

        # Remove NaNs if required
        if self.ignore_nans:
            mask = ~torch.isnan(target)
            target = target[mask]
            output = output[mask]

        device = target.device
        edges = self.bin_edges.to(device)

        # Assign bins: bucketize returns index in [1 .. n_bins]
        bin_idx = torch.bucketize(target, edges, right=False) - 1
        bin_idx = torch.clamp(bin_idx, 0, self.n_quantiles - 1)

        qmse_terms = []

        # -------- Compute QMSE over bins --------
        for k in range(self.n_quantiles):
            mask_k = bin_idx == k
            freq = mask_k.sum()

            if freq == 0:
                continue

            errors_k = (target[mask_k] - output[mask_k]) ** 2
            mse_k = torch.mean(errors_k)

            weight_k = 1.0 / freq.float()
            qmse_terms.append(weight_k * mse_k)

        if len(qmse_terms) == 0:
            return torch.tensor(0.0, device=device)

        return torch.mean(torch.stack(qmse_terms))

    # ------------------------------------------------------------
    # Final Loss = MSE + α * QMSE
    # ------------------------------------------------------------
    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Computes combined MSE and QMSE loss.
        Parameters:
            target (torch.Tensor): Target data.
            output (torch.Tensor): Model output.
        Returns:
            torch.Tensor: Loss value.
        """
        # ------------------------------------------------------------
        # Assert that C = 1 (univariate regression)
        # ------------------------------------------------------------
        if target.ndim == 3:
            B, C, G = target.shape
            assert C == 1, f"Only univariate regression is supported (C=1), got C={C}"
        elif target.ndim == 4:
            B, C, H, W = target.shape
            assert C == 1, f"Only univariate regression is supported (C=1), got C={C}"
        else:
            raise ValueError(
                f"Target tensor must have shape (B, 1, G) or (B, 1, H, W). Got shape: {target.shape}"
            )

        # Same check for output for safety
        if output.ndim == 3:
            B, C, G = output.shape
            assert C == 1, f"Output tensor must have C=1, got C={C}"
        elif output.ndim == 4:
            B, C, H, W = output.shape
            assert C == 1, f"Output tensor must have C=1, got C={C}"
        else:
            raise ValueError(
                f"Output tensor must have shape (B, 1, G) or (B, 1, H, W). Got shape: {output.shape}"
            )

        # ------------------------------------------------------------
        # Convert values below threshold to NaN.
        # ------------------------------------------------------------
        if self.threshold is not None:
            target[target < self.threshold] = np.nan

        # ------------------------------------------------------------
        # Compute loss
        # ------------------------------------------------------------
        mse_val = self.mse(target, output)
        qmse_val = self._compute_qmse(target, output)
        return mse_val + self.alpha * qmse_val


### -------------------------------------------------------------------------------- ###
### -------------------- Per-channel Weighted MSE Loss ---------------------------- ###
class WeightedMseLoss(nn.Module):
    """
    Per-channel weighted Mean Squared Error (MSE) loss for multivariate
    regression.

    Each output channel's MSE is computed independently and the channels are
    combined as a weighted average:

        loss = sum_c ( w_c * MSE_c ) / sum_c ( w_c )     # normalize_weights=True (default)
        loss = sum_c ( w_c * MSE_c )                     # normalize_weights=False

    where ``MSE_c`` is the mean squared error of channel ``c`` (averaged over the
    batch and spatial dimensions) and ``w_c`` is its weight. Use it to prioritise
    a hard channel (e.g. precipitation) over easier ones while keeping a single
    shared backbone. With all weights equal and ``normalize_weights=True`` this
    reduces exactly to ``MseLoss``.

    Parameters
    ----------
    weights : list[float] | tuple[float] | torch.Tensor
        One weight per output channel, in the SAME order as
        ``data.predictands.variables`` in the recipe. Its length must equal the
        number of predictand channels (``model_params.kwargs.in_channels``).
    ignore_nans : bool
        Whether to ignore NaNs in the target domain. NaNs are dropped per
        channel before that channel's mean is taken.
    normalize_weights : bool
        If True (default) divide by ``sum(weights)`` so equal weights recover the
        plain ``MseLoss``; if False use the raw weighted sum.

    YAML usage
    ----------
    Select it in a training recipe under ``model_info.loss_params``. The
    ``module`` stays ``deep4production.deep.loss`` (the class is re-exported
    there), so no path change is needed:

        loss_params:
          name: WeightedMseLoss
          module: deep4production.deep.loss
          kwargs:
            # one weight per predictand channel, in `data.predictands.variables`
            # order. e.g. 8 targets [tas, tasmax, tasmin, pr, hurs, psl, uas, vas]
            # with precipitation (index 3) up-weighted x5:
            weights: [1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0]
            ignore_nans: false
            normalize_weights: true
    """

    def __init__(
        self,
        weights,
        ignore_nans: bool = False,
        normalize_weights: bool = True,
    ) -> None:
        super(WeightedMseLoss, self).__init__()
        self.ignore_nans = ignore_nans
        self.normalize_weights = normalize_weights
        # Stored as a buffer so it follows the module across .to(device) and is
        # saved/restored with the loss state_dict.
        self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32))

    def _per_channel_mse(
        self, target: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        """
        Per-channel mean squared error, shape (C,). Handles (B, C, H, W) or
        (B, C, G) inputs and, when ``ignore_nans`` is set, drops NaNs per
        channel before averaging.
        """
        # --- Handle both spatial (H, W) and flattened (GP) shapes ---
        if target.ndim > 3:  # stack spatial dimensions
            B, C, H, W = target.shape
            target = target.reshape(B, C, -1)  # (B, C, H, W) -> (B, C, H*W)
        if output.ndim > 3:
            B, C, H, W = output.shape
            output = output.reshape(B, C, -1)

        sq_err = (target - output) ** 2  # (B, C, G)
        if self.ignore_nans:
            nan_mask = torch.isnan(target)
            sq_err = torch.where(nan_mask, torch.zeros_like(sq_err), sq_err)
            valid = (~nan_mask).sum(dim=(0, 2)).clamp(min=1)  # (C,) valid count
            return sq_err.sum(dim=(0, 2)) / valid
        return sq_err.mean(dim=(0, 2))  # (C,)

    def _combine(self, per_channel_mse: torch.Tensor) -> torch.Tensor:
        """Weighted combination of the per-channel MSE using ``self.weights``."""
        w = self.weights.to(per_channel_mse.device)
        assert w.numel() == per_channel_mse.numel(), (
            f"{type(self).__name__} got {w.numel()} weights but the target has "
            f"{per_channel_mse.numel()} channels; provide one weight per "
            f"predictand channel."
        )
        weighted = (w * per_channel_mse).sum()
        if self.normalize_weights:
            weighted = weighted / w.sum()
        return weighted

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Computes the per-channel weighted MSE between target and output.
        Parameters:
            target (torch.Tensor): Target data, shape (B, C, H, W) or (B, C, G).
            output (torch.Tensor): Model output, same shape as target.
        Returns:
            torch.Tensor: Scalar loss value.
        """
        return self._combine(self._per_channel_mse(target, output))


### -------------------------------------------------------------------------------- ###
### -------------------- Dynamic Weight Averaging (DWA) MSE Loss ------------------- ###
class DWAWeightedMseLoss(WeightedMseLoss):
    """
    Per-channel MSE with Dynamic Weight Averaging (DWA, Liu et al. 2019,
    "End-to-End Multi-Task Learning with Attention").

    The per-channel weights are NOT fixed: at the end of each epoch they are
    recomputed from how fast each channel's training loss is descending, so a
    slow-progressing channel (e.g. precipitation under multitask competition)
    is automatically up-weighted — no manual weight sweep. With

        L_c(t)   = mean per-channel training MSE at epoch t
        r_c(t-1) = L_c(t-1) / L_c(t-2)              (relative descending rate)
        w_c(t)   = N * softmax_c( r_c(t-1) / T )

    where N is the number of channels and T the temperature (larger T -> more
    uniform weights). For the first two epochs the weights stay uniform (no
    history yet), so DWA reduces to plain MseLoss at the start. The weights sum
    to N (average 1), so the loss scale stays comparable to plain MSE.

    Only *training* forward passes feed the epoch statistics — validation passes
    run under ``torch.no_grad()`` and are skipped, so they never pollute the DWA
    history.

    IMPORTANT — trainer hook
    ------------------------
    The weights are refreshed by ``on_epoch_end()``, which the base
    ``trainer.training_loop`` calls once per epoch (guarded by ``hasattr``).
    Custom trainers that do NOT use the base training loop must call
    ``loss_function.on_epoch_end()`` themselves at the end of each epoch.

    Parameters
    ----------
    num_channels : int
        Number of predictand channels (= ``model_params.kwargs.in_channels``).
    temperature : float
        DWA softmax temperature T. Larger -> weights closer to uniform. The DWA
        paper uses T = 2.0.
    ignore_nans : bool
        Drop NaNs per channel before averaging (see ``WeightedMseLoss``).
    normalize_weights : bool
        Divide the weighted sum by ``sum(weights)``. DWA weights already sum to
        N, so this keeps the loss scale comparable to plain MSE.

    YAML usage
    ----------
    Select it in a training recipe under ``model_info.loss_params``. The
    ``module`` stays ``deep4production.deep.loss`` (the class is re-exported
    there):

        loss_params:
          name: DWAWeightedMseLoss
          module: deep4production.deep.loss
          kwargs:
            num_channels: 8       # one entry per predictand channel
            temperature: 2.0      # DWA softmax temperature
            ignore_nans: false
            normalize_weights: true
    """

    def __init__(
        self,
        num_channels: int,
        temperature: float = 2.0,
        ignore_nans: bool = False,
        normalize_weights: bool = True,
    ) -> None:
        # Start from uniform weights -> plain MSE until two epochs of history
        # are available.
        super(DWAWeightedMseLoss, self).__init__(
            weights=[1.0] * int(num_channels),
            ignore_nans=ignore_nans,
            normalize_weights=normalize_weights,
        )
        self.num_channels = int(num_channels)
        self.temperature = float(temperature)

        # Running accumulation of the current epoch's per-channel training loss.
        self.register_buffer("_epoch_loss_sum", torch.zeros(self.num_channels))
        self.register_buffer("_epoch_count", torch.zeros(()))
        # Mean per-channel loss of the two most recently completed epochs
        # (NaN until that epoch has been seen).
        self.register_buffer(
            "_prev_loss", torch.full((self.num_channels,), float("nan"))
        )
        self.register_buffer(
            "_prev_prev_loss", torch.full((self.num_channels,), float("nan"))
        )

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Computes the DWA-weighted per-channel MSE and, on training passes,
        accumulates the per-channel loss for the epoch's weight update.
        """
        per_channel_mse = self._per_channel_mse(target, output)

        # Accumulate ONLY on training forward passes. Validation runs under
        # torch.no_grad() (grad disabled), so those passes are skipped.
        if torch.is_grad_enabled():
            # Lazily co-locate the DWA buffers with the data the first time, so
            # accumulation stays on-device (no per-batch host<->device sync).
            if self._epoch_loss_sum.device != per_channel_mse.device:
                dev = per_channel_mse.device
                self._epoch_loss_sum = self._epoch_loss_sum.to(dev)
                self._epoch_count = self._epoch_count.to(dev)
                self._prev_loss = self._prev_loss.to(dev)
                self._prev_prev_loss = self._prev_prev_loss.to(dev)
                self.weights = self.weights.to(dev)
            self._epoch_loss_sum += per_channel_mse.detach().float()
            self._epoch_count += 1

        return self._combine(per_channel_mse)

    @torch.no_grad()
    def on_epoch_end(self, epoch: int = None) -> None:
        """
        Refresh the per-channel weights from the epoch's training losses.

        Called once per epoch by the base trainer. Shifts the loss history,
        resets the accumulators and — once two epochs of history exist — sets
        ``w_c = N * softmax(r_c / T)`` with ``r_c = L_c(t-1) / L_c(t-2)``.
        """
        if self._epoch_count.item() == 0:
            return  # no training batches were seen this epoch

        epoch_loss = self._epoch_loss_sum / self._epoch_count  # (C,)

        # Shift history: prev -> prev_prev, current -> prev.
        self._prev_prev_loss = self._prev_loss.clone()
        self._prev_loss = epoch_loss.clone()

        # Reset accumulators for the next epoch.
        self._epoch_loss_sum.zero_()
        self._epoch_count.zero_()

        # Update the weights once two completed epochs allow the ratio; until
        # then they stay uniform (plain MSE).
        if not torch.isnan(self._prev_prev_loss).any():
            # Relative descending rate and DWA weights (sum to N).
            r = self._prev_loss / self._prev_prev_loss.clamp(min=1e-12)
            w = self.num_channels * torch.softmax(r / self.temperature, dim=0)
            self.weights = w.detach().to(self.weights.device)

        # --- Log the per-channel training loss and the weights now in effect ---
        # One host<->device sync per epoch (negligible). The weights logged here
        # are those that will be applied during the NEXT epoch.
        loss_str = ", ".join(
            f"{v:.4e}" for v in self._prev_loss.detach().cpu().tolist()
        )
        w_str = ", ".join(f"{v:.4f}" for v in self.weights.detach().cpu().tolist())
        log.info(
            "DWA | epoch %s | per-channel train MSE: [%s] | weights: [%s]",
            epoch,
            loss_str,
            w_str,
        )
