import numpy as np
import xarray as xr 
from numpy.fft import fft2, fftshift, fftfreq
from scipy.stats import wasserstein_distance as wd
from deep4production.utils.general import get_func_from_string

######### ------------------------------------------------------------------------ #########
# --- HELPER: _mean_over_members ------------------------------------------------------------
def _mean_over_members(da):
    """If DataArray has a 'member' dimension, average over it."""
    if "member" in da.dims:
        return da.mean(dim="member")
    return da
######### ------------------------------------------------------------------------ #########


# --- RMSE ------------------------------------------------------------
def rmse(target, prediction, spatial=False):
    """
    Compute Root Mean Square Error (RMSE) between a target and prediction 
    DataArray, optionally returning a spatial field (map).
    
    - If prediction includes a 'member' dimension, the ensemble mean is used.
    - RMSE is computed over the time dimension.
    
    Parameters
    ----------
    target : xarray.DataArray
        The reference observations or target variable.
    prediction : xarray.DataArray
        Model predictions, possibly with a 'member' dimension.
    spatial : bool
        If True, return RMSE for each spatial point.
        If False, return a global scalar RMSE.
    
    Returns
    -------
    float or xarray.DataArray
        RMSE value(s).
    """

    t = target
    p = prediction
    # Handle ensemble dimension
    if "member" in p.dims:
        p = p.mean(dim="member")
    # Compute squared error
    se = (t - p) ** 2
    # Return spatial field?
    if spatial:
        return np.sqrt(se.mean(dim="time"))
    # Return scalar?
    return float(np.sqrt(se.mean().values))

# --- PSD ------------------------------------------------------------
def _radial_average(array_2d: np.ndarray) -> np.ndarray:
    """
    Compute the radial average of a two-dimensional field.

    Parameters
    ----------
    array_2d : np.ndarray
        Two-dimensional array to average.

    Returns
    -------
    np.ndarray
        Radially averaged profile.
    """
    y, x = np.indices(array_2d.shape)
    center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.hypot(x - center[0], y - center[1]).astype(np.int32)
    tbin = np.bincount(r.ravel(), array_2d.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)

def radially_averaged_power_spectral_density(da, reshape_spatial_dims):
    """
    Compute the power spectral density a 2D spatial field.

    Parameters
    ----------
    da : xarray.DataArray
        Input data. Must contain the dimension `dim`.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        Power spectral densities for x0 and x1.
    """

    if "member" in da.dims:
        da = da.mean(dim="member")

    # --- Transform to 2D ---
    da_vals = da.values
    if len(da.dims) <= 2:
        nt, ng = da_vals.shape
        nx, ny = reshape_spatial_dims
        if nx * ny != ng:
            raise ValueError(f"Cannot reshape: point dimension = {ng}, but nx*ny = {nx*ny}")
        da_vals = da_vals.reshape(nt, ny, nx)

    # --- Transform NaN to 0 values ---
    da_vals = np.nan_to_num(da_vals, nan=0.0)
    
    # --- FFT ---
    fft_da = fftshift(fft2(da_vals, axes=(-2, -1)), axes=(-2, -1))
    power = np.abs(fft_da)**2

    # --- Radial average for each time ---
    psd_list = [_radial_average(power[i]) for i in range(nt)]
    psd_mean = np.mean(psd_list, axis=0)

    # --- Wavenumbers ---
    num_bins = len(psd_list[0])
    wavenumbers = np.arange(num_bins)

    # --- Build coordinates and return DataArray ---
    psd_da = xr.DataArray(
        psd_mean,
        dims=("wavenumber",),
        coords={"wavenumber": wavenumbers},
        name="psd"
    )

    # --- Return ---
    return psd_da


def power_spectral_density(da, dim="time"):
    """
    Compute the 1D Power Spectral Density (PSD) of an xarray DataArray
    along a specified dimension (usually 'time' or a spatial dim like 'point').

    - If a 'member' dimension exists, the ensemble mean is computed first.
    - PSD is computed using the real FFT (rFFT).
    - All non-FFT dimensions are preserved.

    Parameters
    ----------
    da : xarray.DataArray
        Input data. Must contain the dimension `dim`.
    dim : str, optional
        Dimension along which to compute the PSD (default: "time").

    Returns
    -------
    xr.DataArray
        PSD with dimensions ('freq', ...) where the remaining dims match
        the original DataArray except for `dim`.
    """

    ## Ensemble mean 
    if "member" in da.dims:
        da = da.mean(dim="member")

    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' not found in DataArray dims {da.dims}")

    ## Move FFT dimension to axis=0 for NumPy
    da_fft = da.transpose(dim, *[d for d in da.dims if d != dim])

    ## Compute real FFT
    X = np.fft.rfft(da_fft.values, axis=0)
    psd_vals = (np.abs(X) ** 2) / da.sizes[dim]

    ## Frequency coordinates 
    freqs = np.fft.rfftfreq(da.sizes[dim])

    ## Build output dims
    out_dims = ("freq",) + tuple(d for d in da.dims if d != dim)
    out_coords = {"freq": freqs}
    for d in da.dims:
        if d != dim:
            out_coords[d] = da.coords[d]

    ## Return PSD 
    return xr.DataArray(psd_vals, dims=out_dims, coords=out_coords)

# --- Lag-1 autocorrelation ------------------------------------------------------------
def lag1autocorr(da, spatial=False):
    """
    Compute lag-1 autocorrelation along the time axis.
    
    - If 'member' dimension exists, ensemble mean is used.
    - If spatial=True: returns spatial map.
    - If spatial=False: returns global mean scalar.
    
    Parameters
    ----------
    da : xarray.DataArray
        Time series array (time, ...)
    spatial : bool
        If True, return per-gridpoint lag-1 autocorrelation.
        If False, return a scalar mean autocorrelation.
    
    Returns
    -------
    float or xarray.DataArray
        Lag-1 autocorrelation field or scalar.
    """

    # Handle ensemble dimension
    if "member" in da.dims:
        da = da.mean(dim="member")
    # Helper: lag1
    def _lag1(da):
        x1 = da.isel(time=slice(0, -1))
        x2 = da.isel(time=slice(1, None))
        num = ((x1 - x1.mean("time")) * (x2 - x2.mean("time"))).mean("time")
        den = x1.std("time") * x2.std("time")
        return num / den
    # Compute lag1 autocorrelation
    ac = _lag1(da)
    # Return spatial field?
    if spatial:
        return ac
    # Return scalar?
    return float(ac.mean().values)

# --- Quantile ------------------------------------------------------------
def Pxx(da, percentile, spatial=False):
    """
    Compute a percentile of an xarray DataArray.
    If a 'member' dimension exists, compute ensemble mean first.

    Parameters
    ----------
    da : xarray.DataArray
        Dimensions: (time, ...) or (member, time, ...)
    percentile : float
        e.g., 0.02, 0.5, 0.98
    spatial : bool
        - True: return spatial map (lat/lon or point)
        - False: return global scalar

    Returns
    -------
    xarray.DataArray or float
    """

    # Handle ensemble dimension
    if "member" in da.dims:
        da = da.mean(dim="member")
    # Compute quantile over time
    q = da.quantile(percentile, dim="time", skipna=True)
    # Return spatial field?
    if spatial:
        return q
    # Return scalar
    return float(q.mean().values)

# --- P02 ------------------------------------------------------------
def P02(da, spatial=False):
    """ Compute 2nd percentile """

    return Pxx(da, 0.02, spatial=spatial)

# --- P98 ------------------------------------------------------------
def P98(da, spatial=False):
    """ Compute 98th percentile """

    return Pxx(da, 0.98, spatial=spatial)

# --- Median ------------------------------------------------------------
def median(da, spatial=False):
    """ Compute median """

    return Pxx(da, 0.5, spatial=spatial)

# --- Mean ------------------------------------------------------------
def Mean(da, spatial=False):
    """ Compute Mean """

    return da.mean(dim="time") if spatial else da.mean().values

# --- R01 ------------------------------------------------------------
def R01(da, threshold=1.0, percentage=False, spatial=False):
    """
    Count of wet days (precip >= threshold) over the period.

    Parameters
    ----------
    da : xarray.DataArray
        Daily precipitation, dims: (time, ...) or (member, time, ...)
    threshold : float
        Minimum precipitation to consider a wet day (mm)
    spatial : bool
        True: return spatial field
        False: return global scalar

    Returns
    -------
    xarray.DataArray or float
    """
    # Handle ensemble
    if "member" in da.dims:
        da = da.mean(dim="member")

    # Boolean array: True for wet days
    wet = da >= threshold

    # Count wet days along time
    if percentage:
        r01 = wet.mean(dim="time", skipna=True) * 100
    else:
        r01 = wet.sum(dim="time", skipna=True) 

    if spatial:
        return r01
    return float(r01.mean().values)

# --- R20 ------------------------------------------------------------
def R20(da, percentage=False, spatial=False):
    """
    Count of wet days (precip >= threshold) over the period.

    Parameters
    ----------
    da : xarray.DataArray
        Daily precipitation, dims: (time, ...) or (member, time, ...)
    spatial : bool
        True: return spatial field
        False: return global scalar

    Returns
    -------
    xarray.DataArray or float
    """

    # Threshold 20mm
    threshold=20.0

    # Handle ensemble
    if "member" in da.dims:
        da = da.mean(dim="member")

    # Boolean array: True for wet days
    wet = da >= threshold

    # Count wet days along time
    if percentage:
        r01 = wet.mean(dim="time", skipna=True) * 100
    else:
        r01 = wet.sum(dim="time", skipna=True)

    if spatial:
        return r01
    return float(r01.mean().values)

# --- Rx1day ------------------------------------------------------------
def Rx1day(da, threshold=1.0, spatial=False):
    """
    Maximum 1-day precipitation (Rx1day) over the time period.

    Parameters
    ----------
    da : xarray.DataArray
        Daily precipitation, dims: (time, ...) or (member, time, ...)
    threshold : float
        Minimum precipitation to consider a wet day (mm)
    spatial : bool
        True: return spatial field
        False: return global scalar

    Returns
    -------
    xarray.DataArray or float
    """
    if "member" in da.dims:
        da = da.mean(dim="member")

    # Consider only wet days
    wet = da.where(da >= threshold)

    # Maximum over time
    rx1day = wet.max(dim="time", skipna=True)

    if spatial:
        return rx1day
    return float(rx1day.mean().values)


# --- SDII ------------------------------------------------------------
def SDII(da, threshold=1.0, spatial=False):
    """
    Simple Daily Intensity Index: mean precipitation on wet days.

    Parameters
    ----------
    da : xarray.DataArray
        Daily precipitation, dims: (time, ...) or (member, time, ...)
    threshold : float
        Minimum precipitation to consider a wet day (mm)
    spatial : bool
        True: return spatial field
        False: return global scalar

    Returns
    -------
    xarray.DataArray or float
    """
    if "member" in da.dims:
        da = da.mean(dim="member")

    # Select wet days
    wet = da.where(da >= threshold)

    # Mean over time
    sdii = wet.mean(dim="time", skipna=True)

    if spatial:
        return sdii
    return float(sdii.mean().values)


# --- P98Wet ------------------------------------------------------------
def P98Wet(da, threshold=1.0, spatial=False):
    """
    98th percentile of wet-day precipitation.

    Parameters
    ----------
    da : xarray.DataArray
        Daily precipitation, dims: (time, ...) or (member, time, ...)
    threshold : float
        Minimum precipitation to consider a wet day (mm)
    spatial : bool
        True: return spatial field
        False: return global scalar

    Returns
    -------
    xarray.DataArray or float
    """
    if "member" in da.dims:
        da = da.mean(dim="member")

    # Select wet days
    wet = da.where(da >= threshold)

    # Compute 98th percentile over time
    p98 = wet.quantile(0.98, dim="time", skipna=True)

    if spatial:
        return p98
    return float(p98.mean().values)

# --- Bias ------------------------------------------------------------
def bias(target, prediction, index, spatial=False):
    index_fn = get_func_from_string("deep4production.utils.diagnostics", index)
    t = index_fn(target, spatial=True)
    p = index_fn(prediction, spatial=True)
    bias = (p - t)
    if spatial:
        return bias
    return bias.mean().values

# --- Bias Absolute ------------------------------------------------------------
def biasAbs(target, prediction, index, spatial=False):
    index_fn = get_func_from_string("deep4production.utils.diagnostics", index)
    t = index_fn(target, spatial=True)
    p = index_fn(prediction, spatial=True)
    biasAbs = abs(p - t)
    if spatial:
        return biasAbs
    return biasAbs.mean().values

# --- Relative Bias Absolute ------------------------------------------------------------
def relbiasAbs(target, prediction, index, spatial=False):
    index_fn = get_func_from_string("deep4production.utils.diagnostics", index)
    t = index_fn(target, spatial=True)
    p = index_fn(prediction, spatial=True)
    relbiasAbs = abs(p - t) / t * 100
    if spatial:
        return relbiasAbs
    return relbiasAbs.mean().values

# --- Relative Bias ------------------------------------------------------------
def relbias(target, prediction, index, spatial=False):
    index_fn = get_func_from_string("deep4production.utils.diagnostics", index)
    t = index_fn(target, spatial=True)
    p = index_fn(prediction, spatial=True)
    relbias = (p - t) / t * 100
    if spatial:
        return relbias
    return relbias.mean().values


# # --- Wasserstein distance ------------------------------------------------------------
# def wasserstein_dist(
#     ref,
#     sim,
#     by="point",
#     spatial=False,
#     member_mean=True,
#     skipna=True,
#     verbose=False
# ):
#     """
#     Compute Wasserstein (Earth Mover's) distance between two xarray.DataArray objects.

#     Parameters
#     ----------
#     ref : xarray.DataArray
#         Reference data. Dimensions: (time, ...) or (member, time, ...)
#     sim : xarray.DataArray
#         Simulated/predicted data, same dims and coords as ref.
#     by : {'point', 'time', 'global'}
#         - 'point'  : compute distance at each spatial point comparing the time series distributions (recommended for spatial diagnostics)
#         - 'time'   : compute distance at each time comparing spatial distributions across points
#         - 'global' : flatten both arrays (time & space) and compute a single distance
#     spatial : bool
#         Only applies when by == 'point' or by == 'time'.
#         - If by=='point' and spatial==True: return an xarray.DataArray with spatial dims (e.g. point or lat/lon)
#         - If by=='point' and spatial==False: return scalar average of per-point distances
#         - If by=='time' and spatial==True: return xarray.DataArray indexed by time
#         - If by=='time' and spatial==False: return scalar average of per-time distances
#     member_mean : bool
#         If True and the arrays have a 'member' dimension, average over it before computing distances.
#     skipna : bool
#         Whether to ignore NaNs when computing distances at each location/time.
#     verbose : bool
#         Print progress for debugging (useful for many points).

#     Returns
#     -------
#     xarray.DataArray or float
#         Depending on `by` and `spatial`.
#     """

#     # ---- Basic checks ----
#     if not isinstance(ref, xr.DataArray) or not isinstance(sim, xr.DataArray):
#         raise TypeError("ref and sim must be xarray.DataArray objects")
#     # ---- identify dims ----
#     if "time" not in ref.dims:
#         raise ValueError("Input DataArrays must have a 'time' dimension")

#     # ---- Handle ensemble member averaging ----
#     if "member" in sim.dims:
#         da = sim.mean(dim="member")

#     # ---- Get spatial dims ----
#     spatial_dims = [d for d in ref.dims if d != "time"]

#     # ---- Helper to compute WD safely ----
#     def compute_wd(arr1, arr2):
#         a1 = np.asarray(arr1).ravel()
#         a2 = np.asarray(arr2).ravel()
#         if skipna:
#             a1 = a1[~np.isnan(a1)]
#             a2 = a2[~np.isnan(a2)]
#         if a1.size == 0 or a2.size == 0:
#             return np.nan
#         try:
#             return float(wd(a1, a2))
#         except Exception:
#             # fallback simple numpy approach (should rarely happen)
#             a1 = np.sort(a1)
#             a2 = np.sort(a2)
#             # compute 1D empirical CDF difference integral approximate
#             allx = np.unique(np.concatenate([a1, a2]))
#             cu = np.searchsorted(a1, allx, side='right') / float(a1.size)
#             cv = np.searchsorted(a2, allx, side='right') / float(a2.size)
#             widths = np.diff(allx, prepend=allx[0])
#             return float(np.sum(np.abs(cu - cv) * widths))

#     # ----------------- BY = 'global' -----------------
#     if by == "global":
#         arr1 = ref.values.ravel()
#         arr2 = sim.values.ravel()
#         if skipna:
#             arr1 = arr1[~np.isnan(arr1)]
#             arr2 = arr2[~np.isnan(arr2)]
#         if arr1.size == 0 or arr2.size == 0:
#             return np.nan
#         return float(wd(arr1, arr2))

#     # ----------------- BY = 'point' -----------------
#     if by == "point":
#         # stack spatial dims to iterate over points
#         stacked_ref = ref.stack(aux=spatial_dims)
#         stacked_sim = sim.stack(aux=spatial_dims)

#         pts = stacked_ref["aux"].values
#         npts = stacked_ref.sizes["aux"]

#         wd_vals = np.full((npts,), np.nan, dtype=float)

#         for j in range(npts):
#             if verbose and (j % 500 == 0):
#                 print(f"computing WD for point {j+1}/{npts}")
#             series_ref = stacked_ref.isel(aux=j).values  # shape (time,)
#             series_sim = stacked_sim.isel(aux=j).values
#             wd_vals[j] = compute_wd(series_ref, series_sim)

#         # unstack back to original spatial dims
#         wd_da = xr.DataArray(wd_vals, coords={"point": pts}, dims=["point"])
#         # if original spatial dims were e.g. ('lat','lon'), expand back:
#         if len(spatial_dims) > 1:
#             wd_da = wd_da.unstack("point")

#         if spatial:
#             return wd_da
#         else:
#             return float(wd_da.mean().values)

#     # ----------------- BY = 'time' -----------------
#     if by == "time":
#         times = ref["time"].values
#         nt = ref.sizes["time"]
#         wd_vals = np.full((nt,), np.nan, dtype=float)
#         for t in range(nt):
#             if verbose and (t % 50 == 0):
#                 print(f"computing WD for time index {t+1}/{nt}")
#             field_ref = ref.isel(time=t).values.ravel()
#             field_sim = sim.isel(time=t).values.ravel()
#             wd_vals[t] = compute_wd(field_ref, field_sim)
#         wd_da = xr.DataArray(wd_vals, coords={"time": times}, dims=["time"])
#         if spatial:
#             return wd_da
#         else:
#             return float(wd_da.mean().values)

#     raise ValueError("by must be one of {'point','time','global'}")


# # --- wasserstein distance spatial ------------------------------------------------------------
# def wasserstein_dist_spatial(target, prediction, spatial=False):
#     return wasserstein_dist(ref=target, sim=prediction, by="point", spatial=spatial)

# # --- wasserstein distance temporal ------------------------------------------------------------
# def wasserstein_dist_temporal(target, prediction, spatial=False):
#     return wasserstein_dist(ref=target, sim=prediction, by="time", spatial=spatial)