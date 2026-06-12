"""
Gradient-based input attribution (saliency) for trained downscalers.

Purpose
-------
Diagnose *which predictor channels* a trained model relies on, by
differentiating a scalar summary of the predictand (a chosen target variable,
reduced over a user-defined predictand box or gridpoint) with respect to the
network's (normalised) predictor input. The per-date, per-channel attribution
maps let us compare, e.g., a univariate vs a multivariate emulator, or perfect
(UPSRCM) vs imperfect (GCM) predictors, and test hypotheses such as
"the multivariate encoder over-relies on the humidity (hus_*) channels".

Design
------
``Explainer`` subclasses ``downscaler`` so all model / metadata / normalizer /
date / preprocessing machinery is reused unchanged. Only the output location
(``{id_dir}/xai/``) and the forward call differ. Architectures with a
non-standard forward (e.g. the deterministic SongUNet) override
``_attribution_forward`` exactly as the downscaler subclasses override
``downscale``.

Authors
-------
    Jorge Baño-Medina
"""

import os
import numpy as np
import torch
import xarray as xr

from deep4production.core.downscalers.downscaler import downscaler
from deep4production.utils.log import get_logger

log = get_logger("explainer")


class Explainer(downscaler):
    """
    Gradient-based input-attribution explainer.

    Inherits the full ``downscaler`` initialisation (model + metadata +
    normalizers + date pairing + batched preprocessing). Predictions are not
    written; instead ``explain`` writes an attribution Dataset to
    ``{id_dir}/xai/{saving_info['file']}``.
    """

    def __init__(self, id_dir, input_data, **kwargs):
        super().__init__(id_dir=id_dir, input_data=input_data, **kwargs)
        self.id_dir = id_dir
        # Redirect the output from predictions/ to a per-model xai/ directory.
        if self.saving_info is not None:
            self.output_path = f"{id_dir}/xai/{self.saving_info['file']}"
            log.info("Attribution maps will be saved at: %s", self.output_path)
        log.info("Gradient explainer ready")

    # ─────────────────────────────────────────────────────────────────────────
    def _attribution_forward(self, inp_norm, batch_dates):
        """
        Forward pass returning the model output **attached to the autograd
        graph** (no ``inference_mode``/``no_grad``). Mirrors the base
        ``downscaler.downscale`` forward: ``model(inp, f)``. Subclasses override
        for other signatures (see ``ExplainerSongUNetDet``).

        Parameters
        ----------
        inp_norm : torch.Tensor
            (B, C_x, ...) normalised predictor leaf tensor (requires_grad=True).
        batch_dates : list
            Target dates of this batch (used to build forcings if any).

        Returns
        -------
        torch.Tensor : (B, C_y, ...) model output in normalised space.
        """
        f = self._stack_to_device(
            [self._preprocess_forcing_date(d) for d in batch_dates]
        )
        if self.forcing_data is not None and self.norm_f is not None:
            f = self.norm_f(f)
        if self.graph is not None:
            return self.graphPredict(
                x=inp_norm, edge_index=self.edge_index, model=self.model, f=f
            )
        return self.model(inp_norm, f)

    # ─────────────────────────────────────────────────────────────────────────
    def explain(
        self,
        method="gradient",
        target_var=None,
        reduction="mean",
        target_region=None,
        input_space="normalized",
        target_space="physical",
        percent=True,
        batch_size=32,
        verbose=True,
        **kwargs,
    ):
        """
        Compute and save per-date input-attribution maps.

        Parameters
        ----------
        method : str
            "gradient" (saliency = d S / d x). "integrated-gradients" is
            recognised but not yet implemented.
        target_var : str or None
            Predictand whose response is differentiated. Required when the model
            has multiple predictands; defaults to the only channel otherwise.
        reduction : {"mean", "sum"}
            How the selected predictand region is reduced to the scalar S.
            "mean" gives per-gridpoint-average sensitivity; "sum" the total.
        target_region : dict or None
            Predictand box / gridpoint that S is computed over (see
            ``_resolve_target_mask``). None → whole valid domain.
        input_space : {"normalized", "raw"}
            Predictor space the gradient is taken in.
            "normalized" → d S / d x_norm (standardised; channels are unit-
            comparable). "raw" → d S / d x_phys (physical predictor units; the
            normalizer stays in the graph so grads land in raw units, i.e.
            d S / d x_raw = d S / d x_norm / std). With no x-normalizer the two
            coincide.
        target_space : {"physical", "model"}
            Predictand space S is built in. "physical" applies the y-normalizer
            inverse so S is in physical units (e.g. K) — REQUIRED to compare
            models whose predictands are normalised differently (e.g. univariate
            tasmax in raw K vs multivariate tasmax in min-max space). "model"
            uses the raw network output. Note: if the target carries an operator
            (e.g. sqrt on pr/hurs) the physical gradient stays in operator space.
        percent : bool
            Also emit a per-date percentage view (``saliency_percent`` /
            ``channel_contribution_percent``) where each date's |saliency| is
            normalised to sum to 100% across all predictor pixels and variables,
            removing day-to-day magnitude swings (see ``_build_xai_dataset``).
        batch_size : int
            Dates per forward/backward pass. Samples are independent, so a single
            summed backward yields correct per-sample gradients.
        """
        if input_space not in ("normalized", "raw"):
            raise ValueError(
                f"input_space must be 'normalized' or 'raw'; got {input_space!r}"
            )
        if target_space not in ("physical", "model"):
            raise ValueError(
                f"target_space must be 'physical' or 'model'; got {target_space!r}"
            )
        if (self.num_lagged_x or 1) > 1:
            raise NotImplementedError(
                "d4p-explain currently supports single-step predictors "
                "(num_lagged_x == 1)."
            )
        method = (method or "gradient").lower()
        if method in ("integrated-gradients", "integrated_gradients", "ig"):
            # Riemann-sum loop over a straight-line path x0 → x would go here:
            # average the gradients at interpolated inputs and multiply by (x - x0).
            raise NotImplementedError(
                "integrated-gradients is not yet implemented; use method: gradient."
            )
        if method not in ("gradient", "grad", "saliency"):
            raise ValueError(f"Unknown attribution method: {method!r}")

        # Attribution differentiates the model's DIRECT output channels, which
        # equal the predictands only for deterministic (e.g. MSE) regressors.
        # For distributional losses the outputs are distribution parameters and
        # a post-processing maps them to the prediction, so out[:, c] is not the
        # variable — refuse rather than mislead.
        loss_name = (getattr(self, "loss_params", None) or {}).get("name", "")
        if loss_name in ("NLLBerGammaLoss", "NLLGaussianLoss"):
            raise NotImplementedError(
                f"Gradient attribution targets the model's direct output "
                f"channels, but loss '{loss_name}' emits distribution parameters "
                f"(not the predictand). Use a deterministic (MSE) checkpoint, or "
                f"extend the explainer to differentiate the post-processed "
                f"expectation."
            )

        c = self._resolve_target_channel(target_var)
        target_name = self.vars_y[c]
        region_mask = self._resolve_target_mask(target_region, target_name)  # (*spatial_y)
        n_region = int(region_mask.sum().item())
        if n_region == 0:
            raise ValueError("target_region selects no valid gridpoints.")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum'; got {reduction!r}")
        log.info(
            "Explaining '%s' (channel %d) over %d gridpoint(s), reduction=%s, "
            "method=%s",
            target_name, c, n_region, reduction, method,
        )

        self.model.eval()
        T = len(self.target_dates)
        n_batches = (T + batch_size - 1) // batch_size
        sal_batches = []

        for b_idx in range(n_batches):
            i = b_idx * batch_size
            batch_dates = self.target_dates[i : i + batch_size]
            if verbose:
                log.info(
                    "Batch %d/%d: %s → %s (%d dates)",
                    b_idx + 1, n_batches, batch_dates[0], batch_dates[-1],
                    len(batch_dates),
                )

            inp = self._stack_to_device(
                [self._preprocess_single_date(d) for d in batch_dates]
            )  # (B, C_x, ...)

            # Choose the leaf the gradient lands on (grad_leaf) and the tensor
            # the model actually consumes (x_model).
            if self.norm_x is None:
                if input_space == "normalized":
                    log.warning(
                        "No norm_x in metadata: differentiating w.r.t. raw "
                        "inputs (raw == normalized here)."
                    )
                grad_leaf = inp.detach().requires_grad_(True)
                x_model = grad_leaf
            elif input_space == "raw":
                # Leaf is the raw (operator-space) predictor; the affine
                # normalizer stays in the graph so grads arrive in raw units.
                # in_place=False: forward() normalises in place, which would
                # error on a leaf that requires grad.
                grad_leaf = inp.detach().requires_grad_(True)
                x_model = self._apply_norm_x(grad_leaf)
            else:  # "normalized" (default): grad w.r.t. the standardised input
                x_model = self.norm_x(inp).detach().requires_grad_(True)
                grad_leaf = x_model

            out = self._attribution_forward(x_model, batch_dates)  # (B, C_y, *spatial_y)

            # Put S in physical predictand units so models with different
            # y-normalizers are comparable. inverse_transform is affine →
            # differentiable; in_place=False keeps the autograd graph intact.
            if target_space == "physical" and self.norm_y is not None:
                out = self.norm_y.inverse_transform(out, in_place=False)
                if self._ops_y is not None and self._ops_y[c] is not None:
                    log.warning(
                        "target_var '%s' carries an operator; physical-space "
                        "gradient is in operator space (operator inverse is not "
                        "applied differentiably).",
                        target_name,
                    )

            sel = out[:, c]  # (B, *spatial_y)
            masked = sel * region_mask  # broadcast over batch
            per_sample = masked.flatten(start_dim=1).sum(dim=1)  # (B,)
            if reduction == "mean":
                per_sample = per_sample / n_region
            scalar = per_sample.sum()  # samples independent → per-sample grads

            self.model.zero_grad(set_to_none=True)
            scalar.backward()
            sal_batches.append(grad_leaf.grad.detach().to("cpu").numpy())
            leaf_is_model = x_model is grad_leaf
            del inp, out, sel, masked, per_sample, scalar, grad_leaf
            if not leaf_is_model:
                del x_model

        saliency = np.concatenate(sal_batches, axis=0)  # (T, C_x, *spatial_x)
        ds = self._build_xai_dataset(
            saliency, method, target_name, reduction, target_region,
            input_space=input_space, target_space=target_space, percent=percent,
        )

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        log.info("Writing attribution Dataset to %s", self.output_path)
        ds.to_netcdf(self.output_path)
        return ds

    # ─────────────────────────────────────────────────────────────────────────
    def _apply_norm_x(self, x):
        """
        Apply the x-normalizer OUT OF PLACE so it can sit in the autograd graph
        on top of a leaf tensor (forward() normalises in place by default).
        """
        if hasattr(self.norm_x, "transform"):
            return self.norm_x.transform(x, in_place=False)
        return self.norm_x(x)  # plain callable (e.g. test stub)

    # ─────────────────────────────────────────────────────────────────────────
    def _resolve_target_channel(self, target_var):
        """Index of ``target_var`` within ``self.vars_y``."""
        if target_var is None:
            if len(self.vars_y) == 1:
                return 0
            raise ValueError(
                "target_var must be set when the model has multiple predictands: "
                f"{self.vars_y}"
            )
        if target_var not in self.vars_y:
            raise ValueError(
                f"target_var {target_var!r} not in predictands {self.vars_y}"
            )
        return self.vars_y.index(target_var)

    # ─────────────────────────────────────────────────────────────────────────
    def _resolve_target_mask(self, target_region, target_var):
        """
        Build a float mask (on ``self.device``) over the predictand grid marking
        the gridpoints the scalar S is reduced over.

        ``target_region`` schema (all keys optional)::

            type: box | point | all            # default: all when region is None
            # index-based, half-open [start, stop) like numpy slicing:
            i: [i0, i1]   # rows  (H_y / latitude index); int for a point
            j: [j0, j1]   # cols  (W_y / longitude index); int for a point
            # OR geographic, inclusive nearest-gridpoint (needs 1D lats_y/lons_y):
            lat: [lo, hi] # float for a point
            lon: [lo, hi] # float for a point

        The selection is intersected with the template's valid (non-NaN) mask.
        """
        spatial = (
            (self.H_y, self.W_y) if self.transform_to_2D_y else (self.G_y,)
        )
        valid = self._valid_mask_for(target_var, spatial)

        if target_region is None or target_region.get("type", "box") == "all":
            sel = np.ones(spatial, dtype=bool)
        elif self.transform_to_2D_y:
            i0, i1, j0, j1 = self._region_to_indices_2d(target_region)
            sel = np.zeros(spatial, dtype=bool)
            sel[i0:i1, j0:j1] = True
        else:
            g0, g1 = self._region_to_indices_1d(target_region)
            sel = np.zeros(spatial, dtype=bool)
            sel[g0:g1] = True

        region = sel & valid
        return torch.from_numpy(region.astype(np.float32)).to(self.device)

    # ─────────────────────────────────────────────────────────────────────────
    def _valid_mask_for(self, target_var, spatial):
        """Boolean valid (non-NaN) mask from the template, or all-True."""
        if self._template_mask is None:
            return np.ones(spatial, dtype=bool)
        try:
            m = self._template_mask
            name = target_var if target_var in m else list(m.data_vars)[0]
            arr = np.asarray(m[name].values).astype(bool)
            if arr.shape == tuple(spatial):
                return arr
            if arr.size == int(np.prod(spatial)):
                return arr.reshape(spatial)
        except Exception as e:  # pragma: no cover - defensive
            log.warning("Could not derive valid mask from template (%s).", e)
        log.warning("Using all gridpoints as valid.")
        return np.ones(spatial, dtype=bool)

    # ─────────────────────────────────────────────────────────────────────────
    def _region_to_indices_2d(self, region):
        """(i0, i1, j0, j1) half-open index ranges for a 2D predictand grid."""
        rtype = region.get("type", "box")
        if "i" in region or "j" in region:
            i0, i1 = self._index_span(region.get("i"), self.H_y, rtype)
            j0, j1 = self._index_span(region.get("j"), self.W_y, rtype)
            return i0, i1, j0, j1
        if "lat" in region or "lon" in region:
            lats = np.asarray(self.metadata.get("lats_y"))
            lons = np.asarray(self.metadata.get("lons_y"))
            if (
                lats.ndim != 1 or lons.ndim != 1
                or len(lats) != self.H_y or len(lons) != self.W_y
            ):
                raise ValueError(
                    "Geographic target_region requires 1D lats_y (len H_y) and "
                    "lons_y (len W_y) in metadata; use index-based i/j instead."
                )
            i0, i1 = self._geo_span(region.get("lat"), lats, rtype)
            j0, j1 = self._geo_span(region.get("lon"), lons, rtype)
            return i0, i1, j0, j1
        raise ValueError(
            "target_region of type box/point needs i/j (index) or lat/lon "
            "(geographic) keys."
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _region_to_indices_1d(self, region):
        """(g0, g1) half-open index range for a flat (1D) predictand grid."""
        rtype = region.get("type", "box")
        if "g" in region:
            return self._index_span(region.get("g"), self.G_y, rtype)
        raise ValueError(
            "target_region on a 1D (non-2D) predictand grid needs a 'g' "
            "gridpoint-index key."
        )

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _index_span(spec, n, rtype):
        """Half-open [start, stop) from an int (point) or [lo, hi] (box) spec."""
        if spec is None:
            return 0, n
        if rtype == "point" or np.isscalar(spec):
            s = int(spec[0] if isinstance(spec, (list, tuple)) else spec)
            s = max(0, min(s, n - 1))
            return s, s + 1
        lo, hi = int(spec[0]), int(spec[1])
        lo, hi = sorted((lo, hi))
        return max(0, lo), min(hi, n)

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _geo_span(spec, coord1d, rtype):
        """Half-open index range from geographic coordinate(s), inclusive ends."""
        if spec is None:
            return 0, len(coord1d)

        def nearest(v):
            return int(np.argmin(np.abs(coord1d - float(v))))

        if rtype == "point" or np.isscalar(spec):
            s = nearest(spec[0] if isinstance(spec, (list, tuple)) else spec)
            return s, s + 1
        a, b = nearest(spec[0]), nearest(spec[1])
        a, b = sorted((a, b))
        return a, b + 1  # inclusive upper for geographic ranges

    # ─────────────────────────────────────────────────────────────────────────
    def _build_xai_dataset(
        self, saliency, method, target_var, reduction, region,
        input_space="normalized", target_space="physical", percent=True,
    ):
        """
        Wrap (T, C_x, *spatial_x) saliency into an xarray.Dataset.

        Always emits ``saliency`` (signed gradient) and ``channel_importance``
        (per-date mean |saliency| per channel). When ``percent`` is True, also
        emits a magnitude-robust relative view:
          * ``saliency_percent`` — per-date |saliency| renormalised so that the
            sum over ALL predictor pixels AND variables is 100%, i.e. each value
            is the % contribution of that pixel/variable to the date's total
            attribution.
          * ``channel_contribution_percent`` — per-date % of the total
            attribution carried by each predictor variable (spatial sum of
            ``saliency_percent``); sums to 100% across variables per date.
        """
        times = np.array([np.datetime64(d) for d in self.target_dates])
        if self.transform_to_2D_x:
            dims = ("time", "var_x", "y_x", "x_x")
            coords = {
                "time": times,
                "var_x": list(self.vars_x),
                "y_x": np.arange(self.H_x),
                "x_x": np.arange(self.W_x),
            }
            abs_axes = ("y_x", "x_x")
        else:
            dims = ("time", "var_x", "gridpoint_x")
            coords = {
                "time": times,
                "var_x": list(self.vars_x),
                "gridpoint_x": np.arange(self.G_x),
            }
            abs_axes = ("gridpoint_x",)

        ds = xr.Dataset(
            {"saliency": (dims, saliency.astype("float32"))}, coords=coords
        )
        # Headline diagnostic: per-date, per-channel mean |saliency|.
        ds["channel_importance"] = np.abs(ds["saliency"]).mean(dim=abs_axes)

        if percent:
            absS = np.abs(saliency)
            sum_axes = tuple(range(1, absS.ndim))  # over var_x and all spatial
            total = absS.sum(axis=sum_axes, keepdims=True)
            total = np.where(total == 0, np.nan, total)  # avoid 0/0 on null dates
            pct = (absS / total * 100.0).astype("float32")
            ds["saliency_percent"] = (dims, pct)
            ds["channel_contribution_percent"] = ds["saliency_percent"].sum(
                dim=abs_axes
            )

        ds.attrs.update(
            {
                "method": method,
                "target_var": target_var,
                "reduction": reduction,
                "target_region": str(region),
                "input_space": input_space,
                "target_space": target_space,
                "id_dir": str(getattr(self, "id_dir", "")),
                "output_path": str(getattr(self, "output_path", "")),
                "description": (
                    "Gradient of the reduced target-variable response (S) w.r.t. "
                    "the predictor input. 'channel_importance' ranks predictor "
                    "channels by mean |saliency|; 'saliency_percent' gives each "
                    "pixel/variable's % of the per-date total attribution."
                ),
            }
        )
        return ds
