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
(``{id_dir}/outputs/xai/``) and the forward call differ. Architectures with a
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
from deep4production.utils.paths import xai_dir

log = get_logger("explainer")


class Explainer(downscaler):
    """
    Gradient-based input-attribution explainer.

    Inherits the full ``downscaler`` initialisation (model + metadata +
    normalizers + date pairing + batched preprocessing). Predictions are not
    written; instead ``explain`` writes an attribution Dataset to
    ``{id_dir}/outputs/xai/{saving_info['file']}``.
    """

    def __init__(self, id_dir, input_data, **kwargs):
        super().__init__(id_dir=id_dir, input_data=input_data, **kwargs)
        self.id_dir = id_dir
        # Redirect the output from predictions/ to the run's outputs/xai/ dir.
        if self.saving_info is not None:
            self.output_path = f"{xai_dir(id_dir)}/{self.saving_info['file']}"
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
        time_reduction="none",
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
            For ``input_space="raw"`` the |saliency| is first put into
            normalized (scale-comparable) space via σ_c = 1/mul_c, so the
            cross-channel % is not dominated by predictors of small physical
            magnitude (e.g. specific humidity). The signed ``saliency_mean`` and
            ``channel_importance`` stay in raw units.
        time_reduction : {"none", "mean_std"}
            How the time axis is collapsed. "none" (default) writes the full
            per-date stack (``saliency`` / ``channel_importance`` / percent
            views). "mean_std" streams the temporal statistics over the batch
            loop — never materialising the daily stack — and writes only their
            reductions: ``saliency_mean`` (signed E_t[J]), ``saliency_std``,
            ``channel_importance_mean`` (E_t[mean_g|J|]) and, when ``percent``,
            ``saliency_percent_mean`` / ``channel_contribution_percent_mean``.
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
        if time_reduction not in ("none", "mean_std"):
            raise ValueError(
                f"time_reduction must be 'none' or 'mean_std'; got "
                f"{time_reduction!r}"
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
        # time_reduction="none" keeps every per-date map (sal_batches → concat);
        # "mean_std" streams the temporal statistics so the daily stack is never
        # materialised (acc holds running sums, updated per batch).
        reduce_time = time_reduction == "mean_std"
        sal_batches = []
        acc = None

        # Per-channel weights for the relative ("percent") views ONLY. A % over
        # raw-space |J| mixes predictors of different physical scale (e.g. tiny
        # specific-humidity gradients get inflated), so for input_space="raw" we
        # weight each channel by σ_c = 1/mul_c to put |J| in NORMALIZED (scale-
        # comparable) space: σ_c·|∂S/∂x_raw| = |∂S/∂x_norm|. The signed
        # saliency_mean / channel_importance stay in raw units (exact for the
        # shift_decomposition contraction); only the percent allocation changes.
        chan_weights = self._percent_channel_weights(input_space) if percent else None

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
            g = grad_leaf.grad.detach().to("cpu").numpy()  # (B, C_x, *spatial_x)
            if reduce_time:
                acc = self._accumulate_reduction(
                    acc, g, percent=percent, chan_weights=chan_weights
                )
            else:
                sal_batches.append(g)
            leaf_is_model = x_model is grad_leaf
            del inp, out, sel, masked, per_sample, scalar, grad_leaf, g
            if not leaf_is_model:
                del x_model

        if reduce_time:
            ds = self._build_reduced_xai_dataset(
                acc, method, target_name, reduction, target_region,
                input_space=input_space, target_space=target_space,
                percent=percent,
            )
        else:
            saliency = np.concatenate(sal_batches, axis=0)  # (T, C_x, *spatial_x)
            ds = self._build_xai_dataset(
                saliency, method, target_name, reduction, target_region,
                input_space=input_space, target_space=target_space,
                percent=percent, time_reduction=time_reduction,
                chan_weights=chan_weights,
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
    def _percent_channel_weights(self, input_space):
        """
        Per-channel weights ``σ_c`` that put raw-space |saliency| into
        normalized (scale-comparable) space for the relative "percent" views,
        or ``None`` when no weighting is needed.

        The affine normalizer maps ``x_norm = mul·x + add`` so
        ``∂S/∂x_norm = ∂S/∂x_raw / mul``; hence ``σ_c = 1/mul_c`` (= the
        predictor std for mean_std/std). Only meaningful for
        ``input_space="raw"`` with a normalizer present — for "normalized"
        inputs |J| is already comparable (returns None), and with no normalizer
        raw == normalized (None).
        """
        if input_space != "raw" or self.norm_x is None:
            return None
        mul = np.asarray(self.norm_x._norm_mul.detach().cpu(), dtype=np.float64)
        with np.errstate(divide="ignore"):
            w = np.where(mul != 0.0, 1.0 / mul, 0.0)
        return w  # (C,)

    @staticmethod
    def _apply_chan_weights(abs_arr, chan_weights):
        """Multiply |saliency| (…, C, *spatial) by per-channel σ_c on axis 1."""
        if chan_weights is None:
            return abs_arr
        shape = [1] * abs_arr.ndim
        shape[1] = abs_arr.shape[1]  # channel axis
        return abs_arr * np.asarray(chan_weights).reshape(shape)

    # ─────────────────────────────────────────────────────────────────────────
    def _build_xai_dataset(
        self, saliency, method, target_var, reduction, region,
        input_space="normalized", target_space="physical", percent=True,
        time_reduction="none", chan_weights=None,
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
        times = self.target_stamps
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
            # Scale-comparable (normalized-space) |J| for the cross-channel %.
            absS = self._apply_chan_weights(np.abs(saliency), chan_weights)
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
                "time_reduction": time_reduction,
                "id_dir": str(getattr(self, "id_dir", "")),
                "output_path": str(getattr(self, "output_path", "")),
                "description": (
                    "Gradient of the reduced target-variable response (S) w.r.t. "
                    "the predictor input. 'channel_importance' ranks predictor "
                    "channels by mean |saliency| (raw units); 'saliency_percent' "
                    "gives each pixel/variable's % of the per-date total "
                    "attribution, computed in normalized (scale-comparable) space "
                    "for input_space=raw so the cross-channel % is not dominated "
                    "by small-magnitude predictors."
                ),
            }
        )
        return self._attach_predictor_coords(ds)

    # ─────────────────────────────────────────────────────────────────────────
    def _load_predictor_latlon(self):
        """
        (lat, lon) arrays on the PREDICTOR grid, or (None, None) if unavailable.

        Priority: ``saving_info['predictor_template']`` (a NetCDF carrying the
        predictor-grid lat/lon — 1D for a regular grid, 2D for a projected one),
        else the predictor zarr's ``latitudes`` / ``longitudes``.
        """
        tpl = (self.saving_info or {}).get("predictor_template")
        if tpl:
            try:
                t = xr.open_dataset(tpl)
                lat = next(t[n] for n in ("lat", "latitude", "lats") if n in t.variables)
                lon = next(t[n] for n in ("lon", "longitude", "lons") if n in t.variables)
                return np.asarray(lat.values), np.asarray(lon.values)
            except (OSError, StopIteration, KeyError) as e:
                log.warning("Could not read predictor_template '%s' (%s).", tpl, e)
        # Fallback: lat/lon stored alongside the predictor zarr.
        try:
            return (
                np.asarray(self.x[0]["latitudes"][:]),
                np.asarray(self.x[0]["longitudes"][:]),
            )
        except (KeyError, IndexError, TypeError):
            return None, None

    # ─────────────────────────────────────────────────────────────────────────
    def _attach_predictor_coords(self, ds):
        """
        Attach geographic ``lat`` / ``lon`` coordinates to the XAI dataset so the
        saliency maps can be drawn on the predictor grid (plot_saliency reads
        ``da['lat']`` / ``da['lon']``). Handles regular (1D lat/lon) and
        projected (2D lat/lon) grids; warns and leaves the index dims untouched
        if no coordinates can be resolved or their shapes do not match.
        """
        lat, lon = self._load_predictor_latlon()
        if lat is None or lon is None:
            log.warning(
                "No predictor lat/lon available (set saving_info.predictor_template); "
                "saliency written on index dims only."
            )
            return ds

        if "y_x" in ds.dims and "x_x" in ds.dims:
            H, W = ds.sizes["y_x"], ds.sizes["x_x"]
            if lat.ndim == 2 and lat.shape == (H, W) and lon.shape == (H, W):
                return ds.assign_coords(
                    lat=(("y_x", "x_x"), lat), lon=(("y_x", "x_x"), lon)
                )
            if lat.ndim == 1 and lat.size == H and lon.size == W:
                return ds.assign_coords(lat=(("y_x",), lat), lon=(("x_x",), lon))
        elif "gridpoint_x" in ds.dims:
            G = ds.sizes["gridpoint_x"]
            if lat.size == G and lon.size == G:
                return ds.assign_coords(
                    lat=(("gridpoint_x",), lat.ravel()),
                    lon=(("gridpoint_x",), lon.ravel()),
                )

        log.warning(
            "Predictor lat/lon shapes (%s, %s) do not match the saliency grid "
            "%s; coordinates not attached.",
            lat.shape, lon.shape, dict(ds.sizes),
        )
        return ds

    # ─────────────────────────────────────────────────────────────────────────
    def _accumulate_reduction(self, acc, g, percent=True, chan_weights=None):
        """
        Fold one batch of saliency ``g`` (B, C_x, *spatial_x) into the running
        temporal-statistics accumulators (``time_reduction="mean_std"``).

        Tracks running sums of the signed gradient, its square, and the per-date
        channel importance (mean_g|J|); when ``percent`` also the per-date
        relative attribution. ``chan_weights`` (σ_c) scales |J| into normalized
        space for the percent views only — sum_J / sum_J2 / sum_CI stay raw. All
        in float64 for stability over thousands of dates. The daily stack is
        never kept.
        """
        g = g.astype(np.float64)
        spatial_axes = tuple(range(2, g.ndim))  # over the predictor's spatial dims
        absg = np.abs(g)

        if acc is None:
            c_spatial = g.shape[1:]  # (C_x, *spatial_x)
            n_channels = g.shape[1]
            acc = {
                "sum_J": np.zeros(c_spatial, dtype=np.float64),
                "sum_J2": np.zeros(c_spatial, dtype=np.float64),
                "sum_CI": np.zeros(n_channels, dtype=np.float64),
                "n": 0,
                "percent": percent,
            }
            if percent:
                acc["sum_pct"] = np.zeros(c_spatial, dtype=np.float64)
                acc["sum_chan_pct"] = np.zeros(n_channels, dtype=np.float64)
                acc["n_pct"] = 0  # number of non-null dates contributing to pct

        acc["sum_J"] += g.sum(axis=0)
        acc["sum_J2"] += np.square(g).sum(axis=0)
        # per-date channel importance = mean_g|J| over spatial dims → (B, C)
        acc["sum_CI"] += absg.mean(axis=spatial_axes).sum(axis=0)
        acc["n"] += g.shape[0]

        if percent:
            # Scale-comparable (normalized-space) |J| for the cross-channel %.
            absg_w = self._apply_chan_weights(absg, chan_weights)
            # per-date total over var_x AND spatial; null (all-zero) dates are
            # NaN-ed so they neither inflate the sum nor the date count (matches
            # the per-date layout's skipna mean over time).
            sum_axes = tuple(range(1, g.ndim))
            total = absg_w.sum(axis=sum_axes, keepdims=True)  # (B,1,...)
            valid = total[(slice(None),) + (0,) * (g.ndim - 1)] > 0  # (B,)
            total = np.where(total == 0, np.nan, total)
            pct = absg_w / total * 100.0  # (B, C, *spatial); NaN on null dates
            acc["sum_pct"] += np.nansum(pct, axis=0)
            # Channel contribution: spatial NaN-sum per date (null date → 0, so
            # it IS counted, denom = n) — mirrors the per-date layout's
            # channel_contribution_percent = saliency_percent.sum(spatial).
            acc["sum_chan_pct"] += np.nansum(pct, axis=spatial_axes).sum(axis=0)
            acc["n_pct"] += int(valid.sum())

        return acc

    # ─────────────────────────────────────────────────────────────────────────
    def _build_reduced_xai_dataset(
        self, acc, method, target_var, reduction, region,
        input_space="normalized", target_space="physical", percent=True,
    ):
        """
        Build the temporal-reduction XAI Dataset (``time_reduction="mean_std"``)
        from the streamed accumulators.

        Emits (no ``time`` axis):
          * ``saliency_mean``  — signed E_t[J]            (var_x, *spatial)
          * ``saliency_std``   — population std_t[J]      (var_x, *spatial)
          * ``channel_importance_mean`` — E_t[mean_g|J|]  (var_x,)
          * (percent) ``saliency_percent_mean``           (var_x, *spatial)
          * (percent) ``channel_contribution_percent_mean`` (var_x,)
        """
        if self.transform_to_2D_x:
            dims = ("var_x", "y_x", "x_x")
            coords = {
                "var_x": list(self.vars_x),
                "y_x": np.arange(self.H_x),
                "x_x": np.arange(self.W_x),
            }
        else:
            dims = ("var_x", "gridpoint_x")
            coords = {
                "var_x": list(self.vars_x),
                "gridpoint_x": np.arange(self.G_x),
            }

        n = max(acc["n"], 1)
        mean = acc["sum_J"] / n
        # population variance (ddof=0), clamped ≥0 against round-off — matches
        # xarray's default .std("time") on the per-date layout.
        var = np.clip(acc["sum_J2"] / n - mean ** 2, 0.0, None)

        ds = xr.Dataset(
            {
                "saliency_mean": (dims, mean.astype("float32")),
                "saliency_std": (dims, np.sqrt(var).astype("float32")),
                "channel_importance_mean": (
                    ("var_x",), (acc["sum_CI"] / n).astype("float32")
                ),
            },
            coords=coords,
        )

        if percent:
            n_pct = max(acc["n_pct"], 1)
            ds["saliency_percent_mean"] = (
                dims, (acc["sum_pct"] / n_pct).astype("float32")
            )
            ds["channel_contribution_percent_mean"] = (
                ("var_x",), (acc["sum_chan_pct"] / n).astype("float32")
            )

        ds.attrs.update(
            {
                "method": method,
                "target_var": target_var,
                "reduction": reduction,
                "target_region": str(region),
                "input_space": input_space,
                "target_space": target_space,
                "time_reduction": "mean_std",
                "n_dates": int(acc["n"]),
                "id_dir": str(getattr(self, "id_dir", "")),
                "output_path": str(getattr(self, "output_path", "")),
                "description": (
                    "Temporal mean/std of the gradient attribution. "
                    "'saliency_mean' = E_t[dS/dx] (signed, raw units); "
                    "'channel_importance_mean' = E_t[mean_g|dS/dx|] (raw, |.| "
                    "before both reductions, robust to sign cancellation); "
                    "'*_percent_mean' = time-mean of the per-date relative "
                    "attribution, computed in normalized (scale-comparable) space "
                    "for input_space=raw so the cross-channel % is not dominated "
                    "by small-magnitude predictors."
                ),
            }
        )
        return self._attach_predictor_coords(ds)
