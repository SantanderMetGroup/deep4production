"""
Smoke test for the EDM-direct cell (EDM × direct of the 2x2 study).

Exercises the real math of:
  * trainer_edm_direct.trainer_custom.model_backprop — EDM forward on the full
    normalized field + λ(σ)-weighted denoising-score-matching loss + backprop, and
  * downscaler_edm_direct's inherited EDM Heun sampler (downscaler_resdiff.sample),
    which for the direct cell returns the full field (no regressor, no residual).

Builds a tiny EDM-preconditioned SongUNet (build_edm_model) on a small grid and
drives the actual methods on *bare* trainer / downscaler instances
(object.__new__), setting only the attributes those methods touch. No zarr data
or checkpoints; runs on CPU.

Run:  pytest tests/test_edm_direct.py -q
"""

import torch

from deep4production.deep.models.diffusion.edm_precond import build_edm_model
from deep4production.deep.loss import WeightedDenoisingScoreMatchingLoss
from deep4production.core.trainers.trainer_edm_direct import (
    trainer_custom as edm_direct_trainer,
)
from deep4production.core.downscalers.downscaler_edm_direct import (
    downscaler_custom as edm_direct_downscaler,
)


C_Y = 1          # target field channels
C_X = 4          # low-res predictor channels
C_F = 2          # high-res forcing channels
H = W = 16       # target grid
HX = WX = 8      # low-res conditioning grid
B = 2            # batch
SIGMA_DATA = 0.5
NOISE_PARAMS = {"P_mean": -1.2, "P_std": 1.2, "sigma_min": 0.002, "sigma_max": 80.0}


def _tiny_edm_model(cond_high_channels=0):
    """A small EDM-preconditioned SongUNet predicting the denoised full field."""
    return build_edm_model(
        backbone={
            "module": "deep4production.deep.models.unet.song_unet",
            "name": "SongUNet",
            "kwargs": dict(
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
            ),
        },
        sigma_data=SIGMA_DATA,
    )


def _bare_trainer():
    tr = object.__new__(edm_direct_trainer)
    tr.norm_x = tr.norm_y = tr.norm_f = None   # _normalize_inputs → pass-through
    tr.device_type = "cpu"
    tr._scaler = None
    tr._amp_enabled = False
    return tr


def _run_forward(model, data):
    tr = _bare_trainer()
    loss_fn = WeightedDenoisingScoreMatchingLoss(sigma_data=SIGMA_DATA)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss = tr.model_backprop(
        model=model,
        data=data,
        optimizer=optimizer,
        loss_function=loss_fn,
        noise_params=NOISE_PARAMS,
        device="cpu",
        is_this_training=True,
    )
    assert torch.is_tensor(loss) and torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_forward_backprop_direct_field():
    """EDM forward on the full field → λ(σ)-weighted loss → backward; grads finite."""
    torch.manual_seed(0)
    model = _tiny_edm_model(cond_high_channels=0).train()
    x = torch.randn(B, C_X, HX, WX)
    y = torch.randn(B, C_Y, H, W)
    _run_forward(model, (x, y, "N/A"))     # "N/A" sentinel = no forcings


def test_forward_backprop_with_forcings():
    """Same, but with forcings routed into cond_high (cond_high_channels = C_F)."""
    torch.manual_seed(0)
    model = _tiny_edm_model(cond_high_channels=C_F).train()
    x = torch.randn(B, C_X, HX, WX)
    y = torch.randn(B, C_Y, H, W)
    f = torch.randn(B, C_F, H, W)
    _run_forward(model, (x, y, f))


def _bare_downscaler():
    ds = object.__new__(edm_direct_downscaler)
    ds.device = torch.device("cpu")
    ds.num_steps = 5
    ds.sigma_min = NOISE_PARAMS["sigma_min"]
    ds.sigma_max = NOISE_PARAMS["sigma_max"]
    ds.rho = 7.0
    ds.S_churn = 0.0
    ds.S_min = 0.0
    ds.S_max = float("inf")
    ds.S_noise = 1.0
    ds.vars_y = ["pr"]
    ds.H_y, ds.W_y = H, W
    return ds


def test_heun_sampler_direct_field():
    """Inherited EDM Heun sampler returns the full field with correct shape/finite."""
    torch.manual_seed(0)
    model = _tiny_edm_model(cond_high_channels=0).eval()
    ds = _bare_downscaler()
    c_low = torch.randn(B, C_X, HX, WX)
    y = ds.sample(c_low=c_low, c_high=None, model=model)   # full field, not a residual
    assert y.shape == (B, C_Y, H, W)
    assert torch.isfinite(y).all()


def test_heun_sampler_with_forcings():
    """Heun sampler runs with cond_high forcings (C_F channels)."""
    torch.manual_seed(0)
    model = _tiny_edm_model(cond_high_channels=C_F).eval()
    ds = _bare_downscaler()
    c_low = torch.randn(B, C_X, HX, WX)
    f = torch.randn(B, C_F, H, W)
    y = ds.sample(c_low=c_low, c_high=f, model=model)
    assert y.shape == (B, C_Y, H, W)
    assert torch.isfinite(y).all()


if __name__ == "__main__":
    test_forward_backprop_direct_field()
    test_forward_backprop_with_forcings()
    test_heun_sampler_direct_field()
    test_heun_sampler_with_forcings()
    print("All EDM-direct smoke tests passed.")
