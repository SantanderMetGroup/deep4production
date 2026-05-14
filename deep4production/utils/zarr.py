"""
Zarr I/O for deep4production.

Two store layouts are supported:

* **d4p v2** (native; written by ``dataset.to_disk``).
* **anemoi-datasets** (read-only via :class:`AnemoiZarrStore`, exposing the
  d4p v2 contract so the rest of the framework — pydataset, downscaler,
  zarr_inspect — stays format-agnostic).

The single entry point is :func:`open_zarr_store`.
"""

import re
import zarr
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Frequency normalisation: anemoi uses "24h"/"6h"/"1d", d4p uses "1D"/"6H".
# The d4p loop (get_pairs / get_dates_from_yaml) only supports "1D" today, so
# this also acts as a guard — anything that isn't daily is passed through and
# will trip the existing NotImplementedError downstream.
# ─────────────────────────────────────────────────────────────────────────────
_FREQ_RE = re.compile(r"^\s*(\d+)\s*([hdmHDM])\s*$")


def _normalize_frequency(freq):
    if freq is None:
        return None
    if not isinstance(freq, str):
        return freq
    m = _FREQ_RE.match(freq)
    if not m:
        return freq
    n, unit = int(m.group(1)), m.group(2).lower()
    if unit == "h" and n == 24:
        return "1D"
    return f"{n}{unit.upper()}"  # "6h"→"6H", "1d"→"1D", etc.


# ─────────────────────────────────────────────────────────────────────────────
# Anemoi data view: presents the (T, V, E, G) zarr data array as (T, V, G)
# by selecting one ensemble member. d4p code does ``data[i][j][idx_vars]`` —
# this view returns the right shapes for both indexing styles and for
# ``np.array(view)`` materialisation in the load-in-memory path.
# ─────────────────────────────────────────────────────────────────────────────
class _AnemoiDataView:
    def __init__(self, zarr_arr, member=0):
        self._z = zarr_arr
        self._member = member
        # Anemoi data is (T, V, E, G); we expose (T, V, G).
        if zarr_arr.ndim != 4:
            raise ValueError(
                f"AnemoiZarrStore expects 4D data (T, V, E, G); got ndim={zarr_arr.ndim}"
            )
        self.shape = (zarr_arr.shape[0], zarr_arr.shape[1], zarr_arr.shape[3])
        self.dtype = zarr_arr.dtype
        self.ndim = 3

    def __array__(self, dtype=None):
        out = self._z[:, :, self._member, :]
        return out.astype(dtype, copy=False) if dtype is not None else out

    def __getitem__(self, key):
        # Hot path: data[j] → (V, G)
        if isinstance(key, (int, np.integer)):
            return self._z[key, :, self._member, :]
        # Generic fallback (rare in the d4p loop)
        return np.asarray(self.__array__())[key]

    def __len__(self):
        return self.shape[0]


# ─────────────────────────────────────────────────────────────────────────────
# Anemoi → d4p v2 store adapter.
#
# Read-only wrapper around a ``zarr.Group`` produced by anemoi-datasets.
# Re-exposes:
#   ─ obj["data"] / ["dates"] / ["latitudes"] / ["longitudes"]
#   ─ obj["mean"] / ["std"] / ["min"] / ["max"]                ← renamed
#   ─ obj.attrs["variables"] / ["frequency"] / ["shape"] / ["is_regular"]
#                ["H"]/["W"]                                    ← when field_shape present
#                ["variables_metadata"] / ["units"] / ["format_version"]=2
#                ["constant_fields"]                            ← derived from variables_metadata
# ─────────────────────────────────────────────────────────────────────────────
_STAT_KEY_MAP = {"std": "stdev", "min": "minimum", "max": "maximum"}


class AnemoiZarrStore:
    """Read-only adapter exposing an anemoi zarr store as a d4p v2 store."""

    def __init__(self, group, member=0):
        self._z = group
        self._member = member
        self._attrs = self._build_attrs()

    # ── attrs translation ────────────────────────────────────────────────
    def _build_attrs(self):
        z_attrs = dict(self._z.attrs)
        out = {"format_version": 2}

        # Variables: anemoi prefers attrs["variables"] (an ordered iterable of
        # names) and falls back to attrs["name_to_index"] (a {name: idx} dict).
        # We mirror that order. The iterable may be list / tuple / ndarray
        # depending on the zarr serialiser, so we accept anything iterable.
        variables_seq = z_attrs.get("variables")
        if variables_seq is not None:
            try:
                name_to_index = {str(n): i for i, n in enumerate(variables_seq)}
            except TypeError:
                name_to_index = {}
        else:
            name_to_index = dict(z_attrs.get("name_to_index", {}))
        out["variables"] = name_to_index

        # Frequency: "24h" → "1D"
        out["frequency"] = _normalize_frequency(z_attrs.get("frequency"))

        # Variables metadata (same key/structure)
        vmeta = z_attrs.get("variables_metadata", {})
        out["variables_metadata"] = vmeta
        out["constant_fields"] = sorted(
            v
            for v, m in vmeta.items()
            if isinstance(m, dict) and m.get("constant_in_time", False)
        )

        # Units: anemoi nests these in variables_metadata[var]["units"]
        out["units"] = {
            v: (m.get("units") if isinstance(m, dict) else None) or "N/A"
            for v, m in vmeta.items()
        }

        # field_shape → H, W, is_regular
        fs = z_attrs.get("field_shape")
        if fs is not None and len(fs) == 2:
            out["H"] = int(fs[0])
            out["W"] = int(fs[1])
            out["is_regular"] = True
        else:
            out["is_regular"] = False

        # Shape: (T, V, G), dropping anemoi's ensemble axis
        ds_shape = self._z["data"].shape
        out["shape"] = [ds_shape[0], ds_shape[1], ds_shape[3]]
        out["name_dims"] = ["time", "variable", "gridpoint"]

        # Sample counts
        out["num_samples"] = ds_shape[0]
        out["num_samples_yaml"] = ds_shape[0]

        # YAML-period strings (use first/last available date)
        dates = self._z["dates"][:]
        out["date_init_yaml"] = str(dates[0].astype("datetime64[D]"))
        out["date_end_yaml"] = str(dates[-1].astype("datetime64[D]"))

        # Anemoi has no per-variable NaN inventory; expose empties for
        # zarr_inspect.
        n_vars = len(name_to_index)
        out["idx_fixed_nan"] = {v: [] for v in name_to_index}
        out["idx_dynamic_nan"] = {v: [] for v in name_to_index}

        # Pass through anything else useful
        for k in ("resolution", "uuid", "dataset"):
            if k in z_attrs:
                out[k] = z_attrs[k]

        return out

    @property
    def attrs(self):
        return self._attrs

    # ── sub-array translation ────────────────────────────────────────────
    def __getitem__(self, key):
        if key == "data":
            return _AnemoiDataView(self._z["data"], member=self._member)
        if key in ("dates", "latitudes", "longitudes", "mean"):
            return self._z[key]
        if key in _STAT_KEY_MAP:
            return self._z[_STAT_KEY_MAP[key]]
        # Fallback: passthrough (lets advanced callers reach raw anemoi keys)
        return self._z[key]

    def __contains__(self, key):
        if key in (
            "data",
            "dates",
            "latitudes",
            "longitudes",
            "mean",
            "std",
            "min",
            "max",
        ):
            return True
        return key in self._z


# ─────────────────────────────────────────────────────────────────────────────
# Single open() entry point.
# ─────────────────────────────────────────────────────────────────────────────
def _detect_format(group):
    a = group.attrs
    # d4p v2 native stores always stamp format_version=2
    if "format_version" in a:
        return "d4p"
    # anemoi-datasets attrs (checked in order of reliability)
    if "name_to_index" in a:
        return "anemoi"
    # Anemoi layout: attrs["variables"] is an ordered iterable of names
    # (list / tuple / ndarray, depending on serialiser). d4p v2 also has
    # attrs["variables"] but stores it as a {name: idx} dict — so check for
    # non-dict iterables here.
    v = a.get("variables")
    if v is not None and not isinstance(v, dict):
        return "anemoi"
    # Last resort: look for anemoi-specific stat array names or a 4-D data array
    if "stdev" in group or "minimum" in group:
        return "anemoi"
    if "data" in group and group["data"].ndim == 4:
        return "anemoi"
    return "d4p"  # be permissive; let downstream errors surface naturally


def open_zarr_store(path, fmt="auto", member=0, cache_mb=None):
    """
    Open a zarr store and return either the native zarr group (d4p) or an
    :class:`AnemoiZarrStore` (anemoi). Both expose the d4p v2 contract.

    Parameters
    ----------
    path : str
    fmt  : {"auto", "d4p", "anemoi"}
        ``"auto"`` (default) inspects attrs to choose. Set explicitly in YAML
        when you want stricter checking.
    member : int
        For anemoi stores with an ensemble dimension > 1, which member to
        expose. Ignored for d4p stores.
    cache_mb : int or None, default None
        If set, wrap the underlying zarr store in a ``zarr.LRUStoreCache``
        with the given size budget in megabytes. Recently-accessed chunks
        stay in RAM; the rest is evicted. Useful for ``load_in_memory=False``
        runs over slow / shared filesystems. ``None`` disables caching
        (preserves the previous default).
    """
    if cache_mb is not None:
        # Mirrors anemoi-datasets/usage/store.py:92. Requires zarr<=2.18.7
        # — zarr v3 removed LRUStoreCache.
        base = zarr.open(path, mode="r")
        cached_store = zarr.LRUStoreCache(
            base.store, max_size=int(cache_mb) * 1024 * 1024
        )
        group = zarr.open(cached_store, mode="r")
    else:
        group = zarr.open(path, mode="r")
    chosen = _detect_format(group) if fmt == "auto" else fmt
    if chosen == "anemoi":
        return AnemoiZarrStore(group, member=member)
    if chosen == "d4p":
        return group
    raise ValueError(f"Unknown zarr format: {fmt!r}")


# ─────────────────────────────────────────────────────────────────────────────
def zarr_inspect(zarr_path: str, fmt: str = "auto"):
    """
    Inspect and print metadata + statistics from a zarr store.

    Works on both d4p v2 stores and anemoi-datasets stores (via the adapter).

    Parameters
    ----------
    zarr_path : str
        Path to the zarr group.
    fmt : {"auto", "d4p", "anemoi"}, default "auto"
        Format detection mode.
    """
    store = open_zarr_store(zarr_path, fmt=fmt)

    # ── Dates (datetime64[s] sub-array) ──────────────────────────────────────
    dates_raw = store["dates"][:]
    date_init = str(dates_raw[0].astype("datetime64[D]"))
    date_end = str(dates_raw[-1].astype("datetime64[D]"))

    # ── Coordinates ──────────────────────────────────────────────────────────
    lats = store["latitudes"][:]
    lons = store["longitudes"][:]

    # ── Scalar attrs ─────────────────────────────────────────────────────────
    date_init_yaml = store.attrs.get("date_init_yaml", "N/A")
    date_end_yaml = store.attrs.get("date_end_yaml", "N/A")
    num_samples = store.attrs.get("num_samples", "Unknown")
    num_samples_yaml = store.attrs.get("num_samples_yaml", "Unknown")
    is_regular = store.attrs.get("is_regular", "N/A")
    name_dims = store.attrs.get("name_dims", [])
    shape = store.attrs.get("shape", [])

    variables = store.attrs.get("variables", {})  # {name: channel_idx}
    units = store.attrs.get("units", {})
    constant_fields = store.attrs.get("constant_fields", [])
    variables_metadata = store.attrs.get("variables_metadata", {})
    idx_fixed_nan = store.attrs.get("idx_fixed_nan", {})
    idx_dynamic_nan = store.attrs.get("idx_dynamic_nan", {})

    # ── Stats as (C,) sub-arrays ──────────────────────────────────────────────
    means = store["mean"][:]
    stds = store["std"][:]
    mins = store["min"][:]
    maxs = store["max"][:]

    # ── Print ─────────────────────────────────────────────────────────────────
    print("-" * 175)
    print("General information")
    print("-" * 175)
    print(f"{'Date Init (first sample in store):':<40} {date_init}")
    print(f"{'Date End  (last  sample in store):':<40} {date_end}")
    print(f"{'Date Init (requested at creation):':<40} {date_init_yaml}")
    print(f"{'Date End  (requested at creation):':<40} {date_end_yaml}")
    print(f"Number of samples available: {num_samples}/{num_samples_yaml}")

    print()
    print(f"Latitude  range: {np.min(lats):.4f} → {np.max(lats):.4f} degrees")
    print(f"Longitude range: {np.min(lons):.4f} → {np.max(lons):.4f} degrees")
    print(f"Regular grid: {is_regular}")

    print()
    print(f"Dimension names : {name_dims}")
    print(f"Shape           : {shape}")

    if constant_fields:
        print(f"Constant fields : {constant_fields}")

    print()
    print("-" * 175)
    print("Variables summary")
    print("-" * 175)
    header = (
        f"{'Variable':22} | {'Mean':>10} | {'Std':>10} | {'Min':>10} | {'Max':>10} |"
        f" {'Constant':>8} | {'Computed':>8} | {'Fixed NaN pts':>14} | {'Dyn NaN samples':>16} | {'Units':>10}"
    )
    print(header)
    print("-" * 175)

    for var, idx in sorted(variables.items(), key=lambda kv: kv[1]):
        m = f"{means[idx]:.4f}"
        s = f"{stds[idx]:.4f}"
        mn = f"{mins[idx]:.4f}"
        mx = f"{maxs[idx]:.4f}"
        nf = f"{len(idx_fixed_nan.get(var, [])):.0f}"
        nd = f"{len(idx_dynamic_nan.get(var, [])):.0f}"
        unts = units.get(var, "N/A")
        vm = variables_metadata.get(var, {})
        ct = "yes" if vm.get("constant_in_time", False) else "no"
        cf = "yes" if vm.get("computed_forcing", False) else "no"
        print(
            f"{var:22} | {m:>10} | {s:>10} | {mn:>10} | {mx:>10} | {ct:>8} | {cf:>8} | {nf:>14} | {nd:>16} | {unts:>10}"
        )

    print("-" * 175)
