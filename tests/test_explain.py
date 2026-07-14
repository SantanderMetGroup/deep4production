"""
Smoke test for the gradient explainer (d4p-explain).

Drives the real ``ExplainerSongUNetDet.explain`` math on a tiny SongUNet and a
bare explainer instance (object.__new__), setting only the handful of attributes
``explain`` and its helpers touch — predictor preprocessing is monkeypatched so
no zarr / checkpoint / CLI is needed. Runs on CPU in a second or two.

Exercises:
  * autograd path (leaf input → reduced target-region scalar → backward),
  * the target_region box selector,
  * the output Dataset (saliency + channel_importance) shapes/finiteness.

Run:  pytest tests/test_explain.py -q
"""

import numpy as np
import torch

from deep4production.deep.models.unet.song_unet import SongUNet
from deep4production.core.explainers.explainer_song_unet_det import (
    ExplainerSongUNetDet,
)


# --- Tiny fixtures ----------------------------------------------------------
C_X = 4          # low-res predictor channels
C_Y = 2          # predictands (multivariate: e.g. [tasmax, pr])
HX = WX = 8      # low-res conditioning grid
HY = WY = 16     # target grid
VARS_X = ["ua_850", "ta_850", "hus_850", "va_850"]
VARS_Y = ["tasmax", "pr"]


def _tiny_unet():
    return SongUNet(
        in_channels=C_Y,
        cond_low_channels=C_X,
        nf=8,
        ch_mult=[1, 2],
        num_res_blocks=1,
        attn_at_levels=[],
        dropout=0.0,
        cond_upsample="nearest",
        progressive_input=True,
    )


def _bare_explainer(tmp_path):
    """An ExplainerSongUNetDet with only the attributes explain() touches."""
    ex = object.__new__(ExplainerSongUNetDet)
    ex.device = torch.device("cpu")
    ex._cuda = False
    ex.num_lagged_x = 1
    ex.model = _tiny_unet().eval()
    ex.vars_x = list(VARS_X)
    ex.vars_y = list(VARS_Y)
    ex.transform_to_2D_x = True
    ex.transform_to_2D_y = True
    ex.H_x, ex.W_x, ex.G_x = HX, WX, HX * WX
    ex.H_y, ex.W_y, ex.G_y = HY, WY, HY * WY
    ex.norm_x = None
    ex.forcing_data = None
    ex.graph = None
    ex._template_mask = None
    ex.metadata = {}
    ex.id_dir = str(tmp_path)
    ex.output_path = str(tmp_path / "xai.nc")
    ex.target_dates = [np.datetime64("2050-07-15"), np.datetime64("2050-07-16")]
    # Monkeypatch predictor preprocessing → deterministic fake (C_X, HX, WX).
    ex._preprocess_single_date = lambda d: torch.randn(C_X, HX, WX)
    return ex


def test_explain_box_region_produces_finite_saliency(tmp_path):
    torch.manual_seed(0)
    ex = _bare_explainer(tmp_path)

    ds = ex.explain(
        method="gradient",
        target_var="tasmax",
        reduction="mean",
        target_region={"type": "box", "i": [4, 12], "j": [4, 12]},
        batch_size=2,
        verbose=False,
    )

    # Shapes
    assert ds["saliency"].dims == ("time", "var_x", "y_x", "x_x")
    assert ds["saliency"].shape == (2, C_X, HX, WX)
    assert ds["channel_importance"].dims == ("time", "var_x")
    assert ds["channel_importance"].shape == (2, C_X)

    # Finiteness + non-trivial gradient signal
    sal = ds["saliency"].values
    assert np.isfinite(sal).all()
    assert np.abs(sal).sum() > 0.0

    # channel_importance == mean |saliency| over space
    expected = np.abs(sal).mean(axis=(2, 3))
    assert np.allclose(ds["channel_importance"].values, expected)

    # File was written
    assert (tmp_path / "xai.nc").exists()


def test_percent_view_sums_to_100_per_date(tmp_path):
    torch.manual_seed(0)
    ex = _bare_explainer(tmp_path)
    ex.output_path = str(tmp_path / "xai_pct.nc")

    ds = ex.explain(
        method="gradient",
        target_var="tasmax",
        reduction="mean",
        target_region={"type": "all"},
        percent=True,
        batch_size=2,
        verbose=False,
    )
    assert "saliency_percent" in ds and "channel_contribution_percent" in ds
    # Each date's percent map sums to ~100 across (var_x, space).
    per_date_total = ds["saliency_percent"].sum(dim=("var_x", "y_x", "x_x")).values
    assert np.allclose(per_date_total, 100.0, atol=1e-3)
    # channel_contribution_percent sums to ~100 across vars per date.
    chan_total = ds["channel_contribution_percent"].sum(dim="var_x").values
    assert np.allclose(chan_total, 100.0, atol=1e-3)


def test_input_space_raw_scales_by_inverse_std(tmp_path):
    """With an affine norm_x, raw-space grad == normalized-space grad / std."""
    import types

    class _AffineNorm:
        """Minimal stand-in for InputNormalizer: (x - mean) / std, affine."""
        def __init__(self, std):
            self.std = std  # (1, C, 1, 1)
        def __call__(self, x):
            return x / self.std  # mean=0 for simplicity; affine & differentiable

    std = torch.tensor([1.0, 2.0, 4.0, 0.5]).view(1, C_X, 1, 1)

    def run(input_space):
        torch.manual_seed(0)
        ex = _bare_explainer(tmp_path)
        ex.norm_x = _AffineNorm(std)
        ex.output_path = str(tmp_path / f"xai_{input_space}.nc")
        ds = ex.explain(
            method="gradient", target_var="tasmax", reduction="sum",
            target_region={"type": "point", "i": 8, "j": 8},
            input_space=input_space, target_space="model",
            percent=False, batch_size=2, verbose=False,
        )
        return ds["saliency"].values

    g_norm = run("normalized")
    g_raw = run("raw")
    # raw = norm / std (broadcast over the channel axis)
    expected = g_norm / std.view(1, C_X, 1, 1).numpy()
    assert np.allclose(g_raw, expected, atol=1e-4)


def test_explain_single_gridpoint(tmp_path):
    torch.manual_seed(0)
    ex = _bare_explainer(tmp_path)
    ex.output_path = str(tmp_path / "xai_point.nc")

    ds = ex.explain(
        method="gradient",
        target_var="tasmax",
        reduction="mean",
        target_region={"type": "point", "i": 8, "j": 8},
        batch_size=2,
        verbose=False,
    )
    assert ds["saliency"].shape == (2, C_X, HX, WX)
    assert np.isfinite(ds["saliency"].values).all()


def test_integrated_gradients_not_implemented(tmp_path):
    ex = _bare_explainer(tmp_path)
    try:
        ex.explain(method="integrated-gradients", target_var="tasmax")
    except NotImplementedError:
        pass
    else:  # pragma: no cover
        raise AssertionError("integrated-gradients should raise NotImplementedError")


def test_distributional_loss_is_refused(tmp_path):
    ex = _bare_explainer(tmp_path)
    ex.loss_params = {"name": "NLLBerGammaLoss"}
    try:
        ex.explain(method="gradient", target_var="tasmax")
    except NotImplementedError:
        pass
    else:  # pragma: no cover
        raise AssertionError("BerGamma checkpoint should be refused")


def test_target_var_required_for_multivariate(tmp_path):
    ex = _bare_explainer(tmp_path)
    try:
        ex.explain(method="gradient", target_var=None)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("multivariate model without target_var should raise")


if __name__ == "__main__":
    import tempfile, pathlib

    tmp = pathlib.Path(tempfile.mkdtemp())
    test_explain_box_region_produces_finite_saliency(tmp)
    test_explain_single_gridpoint(tmp)
    test_integrated_gradients_not_implemented(tmp)
    test_target_var_required_for_multivariate(tmp)
    print("All d4p-explain smoke tests passed.")
