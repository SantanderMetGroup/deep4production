import torch
import numpy as np
import xarray as xr

# ------------------------------------------------------------------------------------------------------------
def integrated_gradients(
    x,
    model,
    channel_idx=None, # if none gradients computed over all channels
    baseline=None,
    steps=50,
    spatial=False,
    normalize=True,
):
    """
    Integrated Gradients for PyTorch tensors.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with shape (time, C, H, W) or (time, C, P).
    model : callable
        PyTorch model producing outputs of shape (batch, out_channels, ...)
    channel_idx : int or None
        If not None, computes IG for the selected output channel.
    baseline : float or torch.Tensor
        Baseline input, same shape as one x[t]. Default = zeros.
    steps : int
        Number of integration steps.
    spatial : bool
        If True: return (C, H, W) IG fields averaged over time.
        If False: return vector of shape (C,) (avg over time+spatial dims).
    normalize : bool
        Normalize IG per sample so that sum(|IG|) = 1.

    Returns
    -------
    torch.Tensor
        Spatial IG map (C, H, W) or channel IG vector (C,)
    """

    time_dim = x.shape[0]
    x = x.to(torch.float32)

    # --------------------------------------------
    # --- Prepare baseline -----------------------
    if baseline is None:
        baseline = torch.zeros_like(x[0])
    elif isinstance(baseline, (int, float)):
        baseline = torch.full_like(x[0], float(baseline))
    else:
        baseline = baseline.to(x.dtype)

    # -------------------------------------------------
    # --- Compute IG for each time slice separately ---
    ig_list = []
    for t in range(time_dim):
        ## Prepare input data and baseline
        x_t = x[t].clone().detach().requires_grad_(True)
        b_t = baseline

        ## Integrated Gradients loop
        ig_acc = torch.zeros_like(x_t)
        for k in range(1, steps + 1):
            alpha = k / steps
            x_step = (b_t + alpha * (x_t - b_t)).clone().detach().requires_grad_(True)
            out = model(x_step.unsqueeze(0))
            if channel_idx is not None:
                y = out[:, channel_idx].sum()
            else:
                y = out.sum()
            y.backward()
            ig_acc += x_step.grad.detach()
            model.zero_grad(set_to_none=True)

        ig_t = (x_t - b_t) * ig_acc / steps     # (C, H, W)

        ## Normalization
        if normalize:
            s = ig_t.abs().sum()
            if s > 0:
                ig_t = ig_t / s

        ig_list.append(ig_t.detach())

    # ---------------------------------
    # --- Stack along time ------------
    ig_tensor = torch.stack(ig_list, dim=0).cpu()     # shape (time, C, H, W)
    ig_tensor = ig_tensor.mean(dim=0) # Mean over time
    # ------------------------
    # --- Return ------------
    if spatial:
        # average over time → return spatial IG map
        return np.array(ig_tensor)            # (C, H, W)
    else:
        # average over time + spatial dims → return per-channel vector
        reduce_dims = list(range(ig_tensor.ndim))
        reduce_dims.remove(0)  # keep channel dimension (dim=1)
        return np.array(ig_tensor.sum(dim=reduce_dims))  # (C,)