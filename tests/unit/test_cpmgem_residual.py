"""
Smoke test for the CPMGEM-residual cell (sub-VP × residual of the 2x2 study).

Exercises the real math of:
  * trainer_cpmgem_residual.trainer_custom.model_backprop — sub-VP forward on a
    standardized residual + ε-prediction + MSE + backprop, and
  * downscaler_cpmgem_residual.trainer's inherited reverse sub-VP sampler
    (downscaler_cpmgem.sample), plus the un-standardize + add-regressor-mean step.

It builds a tiny plain SongUNet on a small grid and drives the actual methods on
*bare* trainer / downscaler instances (object.__new__), setting only the handful
of attributes those methods touch. No zarr data, checkpoints, or CLI needed; runs
on CPU in a second or two.

Run:  pytest tests/test_cpmgem_residual.py -q
"""

import torch

from deep4production.deep.models.unet.song_unet import SongUNet
from deep4production.deep.loss import MseLoss
from deep4production.core.trainers.trainer_cpmgem_residual import (
    trainer_custom as cpmgem_res_trainer,
)
from deep4production.core.downscalers.downscaler_cpmgem_residual import (
    downscaler_custom as cpmgem_res_downscaler,
)


# --- Tiny shared fixtures (plain helper functions, no pytest fixtures needed) ---
C_Y = 1          # residual / predictand channels
C_X = 4          # low-res predictor channels
C_F = 2          # high-res forcing channels (e.g. orography + land-sea mask)
H = W = 16       # target grid
HX = WX = 8      # low-res conditioning grid
B = 2            # batch
NOISE_PARAMS = {"beta_min": 0.1, "beta_max": 20.0, "t_min": 1e-5}


def _tiny_unet(cond_high_channels=C_Y):
    """A small plain SongUNet predicting ε.

    cond_high_channels = C_Y for the regressor-mean-only case, or C_Y + C_F when
    forcings are concatenated onto the regressor mean (cond_high = [ŷ_det, f]).
    """
    return SongUNet(
        in_channels=C_Y,
        cond_low_channels=C_X,
        cond_high_channels=cond_high_channels,
        nf=8,
        ch_mult=[1, 2],
        num_res_blocks=1,
        attn_at_levels=[],
        dropout=0.0,
        cond_upsample="nearest",
        progressive_input=True,
    )


def _res_stats():
    """Non-trivial per-variable standardization stats (1, C_Y, 1, 1)."""
    mean = torch.full((1, C_Y, 1, 1), 0.05)
    std = torch.full((1, C_Y, 1, 1), 0.20)  # residual std << 1, the whole point
    return mean, std


def test_forward_backprop_runs_with_finite_grads():
    """sub-VP forward on standardized residual → ε-MSE → backward; grads finite."""
    torch.manual_seed(0)
    model = _tiny_unet().train()
    loss_fn = MseLoss(ignore_nans=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    mean, std = _res_stats()

    # Bare trainer with only the attributes model_backprop touches.
    tr = object.__new__(cpmgem_res_trainer)
    tr.standardize_residual = True
    tr._res_mean, tr._res_std = mean, std
    tr.norm_x = None
    tr.device_type = "cpu"
    tr._scaler = None
    tr._amp_enabled = False

    # (residual, c_low, c_high) as served by pydataset_resdiff.
    r = torch.randn(B, C_Y, H, W) * std + mean      # realistic small-variance residual
    c_low = torch.randn(B, C_X, HX, WX)
    c_high = torch.randn(B, C_Y, H, W)              # regressor mean ŷ_det

    loss = tr.model_backprop(
        model=model,
        data=(r, c_low, c_high),
        optimizer=optimizer,
        loss_function=loss_fn,
        noise_params=NOISE_PARAMS,
        device="cpu",
        is_this_training=True,
    )

    assert torch.is_tensor(loss) and torch.isfinite(loss), "loss must be a finite tensor"

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients were populated by backward()"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradients"


def test_standardization_makes_residual_unit_variance():
    """The trainer's standardization should map the residual to ~unit variance."""
    mean, std = _res_stats()
    r = torch.randn(10_000, C_Y, 1, 1) * std + mean
    r_std = (r - mean) / std
    assert abs(float(r_std.std()) - 1.0) < 0.05
    assert abs(float(r_std.mean())) < 0.05


def test_reverse_sampler_and_combine():
    """Inherited reverse sub-VP chain returns correct shape/finite; combine runs."""
    torch.manual_seed(0)
    model = _tiny_unet().eval()
    mean, std = _res_stats()

    # Bare downscaler with only the attributes sample()/_reverse_step/_sde_coeffs use.
    ds = object.__new__(cpmgem_res_downscaler)
    ds.device = torch.device("cpu")
    ds.beta_min, ds.beta_max = NOISE_PARAMS["beta_min"], NOISE_PARAMS["beta_max"]
    ds.t_min = 1e-3
    ds.num_steps = 5            # tiny chain — we only check shape/finiteness
    ds.denoise = True
    ds.vars_y = ["pr"]          # len == C_Y
    ds.H_y, ds.W_y = H, W
    ds._res_mean = mean.to(ds.device)
    ds._res_std = std.to(ds.device)

    c_low = torch.randn(B, C_X, HX, WX)
    c_high = torch.randn(B, C_Y, H, W)              # regressor mean → cond_high

    # Real inherited reverse sub-VP sampler (returns the standardized residual).
    r_std = ds.sample(x_cond=c_low, model=model, f_cond=c_high)
    assert r_std.shape == (B, C_Y, H, W)
    assert torch.isfinite(r_std).all(), "reverse chain produced non-finite values"

    # Un-standardize + add regressor mean → normalized y-space (downscale() math).
    r_hat = r_std * ds._res_std + ds._res_mean
    p = c_high + r_hat
    assert p.shape == (B, C_Y, H, W)
    assert torch.isfinite(p).all()


def test_forward_with_forcings_concatenated_in_cond_high():
    """cond_high = [ŷ_det, f]: forward backprops with the wider cond_high stream.

    In the residual cells pydataset_resdiff concatenates the forcing onto the
    regressor mean before the trainer sees it, so model_backprop just receives a
    (C_Y + C_F)-channel c_high. Here we build that combined stream directly.
    """
    torch.manual_seed(0)
    model = _tiny_unet(cond_high_channels=C_Y + C_F).train()
    loss_fn = MseLoss(ignore_nans=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    mean, std = _res_stats()

    tr = object.__new__(cpmgem_res_trainer)
    tr.standardize_residual = True
    tr._res_mean, tr._res_std = mean, std
    tr.norm_x = None
    tr.device_type = "cpu"
    tr._scaler = None
    tr._amp_enabled = False

    r = torch.randn(B, C_Y, H, W) * std + mean
    c_low = torch.randn(B, C_X, HX, WX)
    y_det = torch.randn(B, C_Y, H, W)
    f = torch.randn(B, C_F, H, W)
    c_high = torch.cat([y_det, f], dim=1)          # regressor mean first, forcing second
    assert c_high.shape[1] == C_Y + C_F

    loss = tr.model_backprop(
        model=model,
        data=(r, c_low, c_high),
        optimizer=optimizer,
        loss_function=loss_fn,
        noise_params=NOISE_PARAMS,
        device="cpu",
        is_this_training=True,
    )
    assert torch.is_tensor(loss) and torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_reverse_with_forcings_concatenated_in_cond_high():
    """Reverse sub-VP chain runs with cond_high = [ŷ_det, f] (C_Y + C_F channels)."""
    torch.manual_seed(0)
    model = _tiny_unet(cond_high_channels=C_Y + C_F).eval()
    mean, std = _res_stats()

    ds = object.__new__(cpmgem_res_downscaler)
    ds.device = torch.device("cpu")
    ds.beta_min, ds.beta_max = NOISE_PARAMS["beta_min"], NOISE_PARAMS["beta_max"]
    ds.t_min = 1e-3
    ds.num_steps = 5
    ds.denoise = True
    ds.vars_y = ["pr"]
    ds.H_y, ds.W_y = H, W
    ds._res_mean = mean.to(ds.device)
    ds._res_std = std.to(ds.device)

    c_low = torch.randn(B, C_X, HX, WX)
    y_det = torch.randn(B, C_Y, H, W)
    f = torch.randn(B, C_F, H, W)
    c_high = torch.cat([y_det, f], dim=1)          # same order as the downscaler concat

    r_std = ds.sample(x_cond=c_low, model=model, f_cond=c_high)
    assert r_std.shape == (B, C_Y, H, W)           # residual stays C_Y, not C_Y + C_F
    assert torch.isfinite(r_std).all()

    # Combine uses only the regressor-mean part of cond_high (y_det), not forcings.
    r_hat = r_std * ds._res_std + ds._res_mean
    p = y_det + r_hat
    assert p.shape == (B, C_Y, H, W) and torch.isfinite(p).all()


if __name__ == "__main__":
    test_forward_backprop_runs_with_finite_grads()
    test_standardization_makes_residual_unit_variance()
    test_reverse_sampler_and_combine()
    test_forward_with_forcings_concatenated_in_cond_high()
    test_reverse_with_forcings_concatenated_in_cond_high()
    print("All CPMGEM-residual smoke tests passed.")
