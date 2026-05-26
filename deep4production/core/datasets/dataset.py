import os
import glob
import numcodecs
import cftime
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from torch.utils.data import Dataset

## Deep4production
from deep4production.utils.forcings import (
    compute_julian_day,
    compute_sincos_coords,
    compute_toa_solar_radiation,
)
from deep4production.utils.trans import xarray_to_numpy
from deep4production.utils.general import is_grid_regular
from deep4production.utils.imputers import d4dimputers
from deep4production.utils.log import get_logger

log = get_logger("dataset")

# Variables that are always computed analytically from coordinates / date.
# constant_in_time=True  → identical across all time steps (e.g. sin(lat))
# constant_in_time=False → changes per time step but is never read from NetCDF
_KNOWN_COMPUTED_FORCINGS = {
    "sin_lat": {"computed_forcing": True, "constant_in_time": True},
    "cos_lat": {"computed_forcing": True, "constant_in_time": True},
    "sin_lon": {"computed_forcing": True, "constant_in_time": True},
    "cos_lon": {"computed_forcing": True, "constant_in_time": True},
    "sin_julian_day": {"computed_forcing": True, "constant_in_time": False},
    "cos_julian_day": {"computed_forcing": True, "constant_in_time": False},
    "toa_solar_radiation": {"computed_forcing": True, "constant_in_time": False},
}


########################################################################################################
class dataset(Dataset):
    """
    Dataset class for loading and processing NetCDF climate data.
    Purpose: Handles spatial, temporal, and variable selection, imputation, and statistics
    computation for deep learning workflows. Produces Zarr format v2 stores.
    Parameters:
        date_init (str): Start date for dataset.
        date_end (str): End date for dataset.
        freq (str): Temporal frequency (e.g., 'D', 'M').
        data (dict): Dataset configuration (paths, variables, imputer, etc).
    """

    def __init__(self, date_init, date_end, freq, data):
        # --- GENERAL INFO ------
        self.data = data
        self.variables = data["vars"]
        self.num_vars = len(self.variables)

        raw_sources = self.data["paths"]
        self.source_files = []
        for source in raw_sources:
            if "*" in source:
                matched_files = sorted(glob.glob(source))
                if not matched_files:
                    log.warning("No files matched pattern: %s", source)
                self.source_files.extend(matched_files)
            elif os.path.isfile(source):
                self.source_files.append(source)
            else:
                log.warning("File not found: %s", source)
        self.num_sources = len(self.source_files)
        log.info("Found %d source NetCDF files.", self.num_sources)

        # --- TEMPORAL INFORMATION ------
        self.date_init = pd.to_datetime(date_init)
        self.date_end = pd.to_datetime(date_end)
        self.freq = freq
        dates_yaml = pd.date_range(
            start=self.date_init, end=self.date_end, freq=self.freq
        ).to_numpy()
        self.num_samples_yaml = len(dates_yaml)
        available_dates = self.get_available_dates_in_sources(self.source_files)
        self.dates = np.array([d for d in dates_yaml if d in available_dates])
        self.num_samples = len(self.dates)
        msg = "Found %d of %d requested samples from the YAML period in the source NetCDF files."
        if self.num_samples == self.num_samples_yaml:
            log.info(msg, self.num_samples, self.num_samples_yaml)
        else:
            log.warning(msg, self.num_samples, self.num_samples_yaml)

        # --- SPATIAL INFORMATION ------
        with xr.open_dataset(self.source_files[0]) as temp:
            self.spatial_dims, self.is_regular = self.get_spatial_dims(temp)
            if self.is_regular:
                self.H = len(temp[self.spatial_dims[0]].values)
                self.W = len(temp[self.spatial_dims[1]].values)
                temp = temp.stack(point=self.spatial_dims)
            self.number_gridpoints = len(temp[self.spatial_dims[0]].values)
            self.lat = temp.lat.values
            self.lon = temp.lon.values
            del temp

        # --- IMPUTE NANS (if any) ------
        self.imputer = data.get("imputer", None)

    # ---------------------------------------------------------------
    def get_spatial_dims(self, dataset):
        """
        Determines spatial dimensions and grid regularity from an xarray dataset.
        Parameters:
            dataset (xarray.Dataset): Dataset to inspect.
        Returns:
            tuple: (spatial_dims, is_regular)
        """
        if {"x", "y"}.issubset(dataset.dims):
            spatial_dims = ["y", "x"]
        elif {"lat", "lon"}.issubset(dataset.dims):
            spatial_dims = ["lat", "lon"]
        else:
            spatial_dims = ["point"]
        grid_type = is_grid_regular(dataset)
        return spatial_dims, grid_type

    # ---------------------------------------------------------------
    def get_available_dates_in_sources(self, paths):
        """
        Extracts available dates from NetCDF source files.
        Parameters:
            paths (list): List of NetCDF file paths.
        Returns:
            np.ndarray: Array of available dates.
        """
        available_dates = []
        for p in paths:
            try:
                with xr.open_dataset(p) as ds:
                    times = ds["time"].values
                    times_str = [str(t) for t in times]
                    available_dates.append(times_str)
            except Exception as e:
                log.warning("Could not read %s: %s", p, e)
        return np.array(np.concatenate(available_dates), dtype="datetime64[ns]")

    # ---------------------------------------------------------------
    def compute_stats_per_channel(self, zarr_group):
        """
        Single-pass NaN-aware mean, std, min, max per channel.

        Uses sum-of-squares accumulation in float64 (var = E[X²] − E[X]²,
        Bessel-corrected). Stable for climate magnitudes; one pass over the
        store instead of three.

        Parameters:
            zarr_group (zarr.Group): Opened zarr group (format v2).
        Returns:
            tuple: (mean, std, min, max) — float32 arrays of shape (C,).
        """
        z = zarr_group["data"]
        S, C, _ = z.shape
        n = np.zeros(C, dtype=np.int64)
        sum1 = np.zeros(C, dtype=np.float64)
        sum2 = np.zeros(C, dtype=np.float64)
        min_vals = np.full(C, np.inf, dtype=np.float32)
        max_vals = np.full(C, -np.inf, dtype=np.float32)
        for i in range(S):
            x = z[i]  # (C, G) float32
            nan_mask = np.isnan(x)
            valid = ~nan_mask
            x_safe = np.where(nan_mask, 0.0, x.astype(np.float64))
            sum1 += x_safe.sum(axis=1)
            sum2 += (x_safe * x_safe).sum(axis=1)
            n += valid.sum(axis=1)
            # NaN-safe per-channel min/max
            min_vals = np.minimum(min_vals, np.where(nan_mask, np.inf, x).min(axis=1))
            max_vals = np.maximum(max_vals, np.where(nan_mask, -np.inf, x).max(axis=1))
        n_safe = np.maximum(n, 1)
        mean = sum1 / n_safe
        var = (sum2 - n * mean**2) / np.maximum(n - 1, 1)
        std = np.sqrt(np.maximum(var, 0.0))
        return mean.astype(np.float32), std.astype(np.float32), min_vals, max_vals

    # ---------------------------------------------------------------
    def count_nans(self, zarr_group):
        """
        Counts NaNs per channel and gridpoint in the data sub-array.
        Parameters:
            zarr_group (zarr.Group): Opened zarr group (format v2).
        Returns:
            tuple: (fixed_nan, dynamic_nan)
        """
        z = zarr_group["data"]
        S, C, G = z.shape
        nan_count = np.zeros((C, G), dtype=np.int64)
        dynamic_nan = {c: [] for c in range(C)}
        for s in range(S):
            x = z[s]
            nan_mask = np.isnan(x)
            nan_count += nan_mask.astype(np.int64)
            for c in range(C):
                gp_idx = np.where(nan_mask[c])[0]
                for g in gp_idx:
                    dynamic_nan[c].append([s, int(g)])
        fixed_nan = {c: np.where(nan_count[c] == S)[0] for c in range(C)}
        for c in range(C):
            fixed_set = set(fixed_nan[c])
            dynamic_nan[c] = [
                pair for pair in dynamic_nan[c] if pair[1] not in fixed_set
            ]
        return fixed_nan, dynamic_nan

    # ---------------------------------------------------------------
    def impute_nans(self, zarr_group, zarr_attrs, lats, lons):
        """
        Imputes dynamic NaNs in the data sub-array using specified imputer functions.

        For each (variable, timestep), the row is read once, all NaN gridpoints
        in that row are imputed against that row, and the row is written back
        once. Avoids the previous behaviour of re-reading the full row from disk
        for every NaN gridpoint.

        Parameters:
            zarr_group (zarr.Group): Opened zarr group (format v2).
            zarr_attrs (dict): Dynamic NaN indices keyed by variable name.
            lats (np.ndarray): Latitude values.
            lons (np.ndarray): Longitude values.
        Returns:
            dict: Updated zarr_attrs with imputed entries cleared.
        """
        variables_map = zarr_group.attrs["variables"]
        data_arr = zarr_group["data"]
        for var in self.variables:
            idx_var = variables_map[var]
            imputer_default = self.imputer.get("default")
            imputer_selected = self.imputer.get(var, imputer_default)
            imputer_name = imputer_selected["name"]
            kwargs_imputer = {k: v for k, v in imputer_selected.items() if k != "name"}
            dyn_list = zarr_attrs.get(var, [])
            if not dyn_list:
                log.info("[%s] No dynamic NaNs found, skipping imputation.", var)
                continue

            log.info(
                "[%s] Starting dynamic NaN imputation using '%s'", var, imputer_name
            )
            # Group gridpoints by timestep so each (t, idx_var) row is read once.
            gps_by_t = {}
            for t, gp in dyn_list:
                gps_by_t.setdefault(int(t), []).append(int(gp))

            for t, gps in gps_by_t.items():
                row = data_arr[t, idx_var, :]  # one read
                log.debug(
                    "[%s] t=%d (%s): imputing %d gridpoint(s)",
                    var,
                    t,
                    self.dates[t],
                    len(gps),
                )
                for gp in gps:
                    imp = d4dimputers(
                        data=row,
                        lat_gp=lats[gp],
                        lon_gp=lons[gp],
                        lats_ref=lats,
                        lons_ref=lons,
                    )
                    row[gp] = getattr(imp, imputer_name)(**kwargs_imputer)
                data_arr[t, idx_var, :] = row  # one write
            zarr_attrs[var] = []
        return zarr_attrs

    # ---------------------------------------------------------------
    def get_units(self, ds, var):
        """
        Retrieves units attribute for a variable from an xarray dataset.
        Parameters:
            ds (xarray.Dataset): Dataset.
            var (str): Variable name.
        Returns:
            str: Units string.
        """
        units = ds[var].attrs.get("units", "N/A")
        if units == "N/A":
            log.warning("No units attribute found for variable '%s'", var)
        return units

    # ---------------------------------------------------------------
    def to_disk(self, zarr_path):
        """
        Saves the processed dataset to disk as a Zarr v2 store.

        Store layout
        ------------
        zarr_group/
          data/          (T, C, G) float32  — all variable data, chunks (1, C, G)
          dates/         (T,)      datetime64[s]
          latitudes/     (G,)      float32
          longitudes/    (G,)      float32
          mean/          (C,)      float32  — per-channel statistics (NaN-aware)
          std/           (C,)      float32
          min/           (C,)      float32
          max/           (C,)      float32

        Group attributes (no large arrays)
        -----------------------------------
        format_version (int=2), variables {name:idx}, units {var:str},
        frequency, shape [T,C,G], is_regular, H/W (if regular),
        constant_fields (list[str]), variables_metadata {var:{computed_forcing,constant_in_time}},
        idx_fixed_nan, idx_dynamic_nan, date_init_yaml, date_end_yaml,
        num_samples, num_samples_yaml, name_dims.

        Parameters:
            zarr_path (str): Output path for the Zarr store.
        Returns:
            str: Confirmation message.
        """
        blk = numcodecs.Blosc(cname="zstd", clevel=5)

        # --- Open zarr GROUP ---
        zarr_group = zarr.open_group(zarr_path, mode="w")

        # --- Create data sub-array ---
        data_arr = zarr_group.create_dataset(
            "data",
            shape=(self.num_samples, self.num_vars, self.number_gridpoints),
            chunks=(1, self.num_vars, self.number_gridpoints),
            dtype="float32",
            compressor=blk,
            fill_value=np.nan,
        )

        # --- Group-level attributes (no large arrays stored here) ---
        zarr_group.attrs["format_version"] = 2
        zarr_group.attrs["date_init_yaml"] = str(self.date_init)
        zarr_group.attrs["date_end_yaml"] = str(self.date_end)
        zarr_group.attrs["num_samples"] = self.num_samples
        zarr_group.attrs["num_samples_yaml"] = self.num_samples_yaml
        zarr_group.attrs["frequency"] = self.freq
        zarr_group.attrs["variables"] = {
            var: idx for idx, var in enumerate(self.variables)
        }
        zarr_group.attrs["name_dims"] = ["time", "variable", "gridpoint"]
        zarr_group.attrs["shape"] = [
            self.num_samples,
            self.num_vars,
            self.number_gridpoints,
        ]
        zarr_group.attrs["is_regular"] = self.is_regular
        if self.is_regular:
            zarr_group.attrs["H"] = self.H
            zarr_group.attrs["W"] = self.W

        # --- Sub-arrays: dates (datetime64[s], native), latitudes, longitudes ---
        dates_dt = np.array(self.dates, dtype="datetime64[s]")
        zarr_group.create_dataset(
            "dates",
            data=dates_dt,
            chunks=(len(dates_dt),),
            dtype="datetime64[s]",
            compressor=blk,
        )
        zarr_group.create_dataset(
            "latitudes",
            data=self.lat.astype(np.float32),
            chunks=(len(self.lat),),
            dtype="float32",
            compressor=blk,
        )
        zarr_group.create_dataset(
            "longitudes",
            data=self.lon.astype(np.float32),
            chunks=(len(self.lon),),
            dtype="float32",
            compressor=blk,
        )

        # --- Write data from NetCDF sources ---
        # constant_vars: variables with no time dimension (e.g. orography)
        constant_vars = set()
        units_dict = {}

        for source in self.source_files:
            x = xr.open_dataset(source)
            for var in x.data_vars:
                if var not in self.variables:
                    log.debug(
                        "Skipping variable %s in %s, not in target variable list.",
                        var,
                        source,
                    )
                    continue

                log.info("Variable %s from %s matches target variables.", var, source)
                idx_var = zarr_group.attrs["variables"][var]
                x_ = x[[var]]
                units_dict[var] = self.get_units(ds=x_, var=var)

                if "time" in x_.dims:
                    avail_dates = x_.time.values.astype("datetime64[ns]")
                    matching_dates = np.intersect1d(self.dates, avail_dates)
                    if len(matching_dates) == 0:
                        log.warning(
                            "No dates in %s match the requested period; skipping.",
                            source,
                        )
                        continue
                    idx_samples = [
                        np.where(self.dates == t)[0][0] for t in matching_dates
                    ]
                    if isinstance(x_.time.values[0], cftime.DatetimeNoLeap):
                        x_ = x_.convert_calendar("standard")
                    x_ = x_.sel(time=matching_dates)
                    if self.is_regular:
                        x_ = x_.stack(point=self.spatial_dims)
                    xdata = xarray_to_numpy(x_).astype(np.float32)
                    x_.close()
                    del x_
                    for i, t_idx in enumerate(idx_samples):
                        data_arr[t_idx, idx_var, :] = xdata[i]
                else:
                    # No time dimension → constant field (e.g. orography)
                    constant_vars.add(var)
                    if self.is_regular:
                        x_ = x_.stack(point=self.spatial_dims)
                    xdata = xarray_to_numpy(x_).astype(np.float32)
                    x_.close()
                    del x_
                    data_arr[:, idx_var, :] = np.tile(xdata, (self.num_samples, 1))

            x.close()
            del x

        zarr_group.attrs["units"] = units_dict

        # --- Computed forcings (analytical, no NetCDF read needed) ---
        for var in self.variables:
            idx_var = zarr_group.attrs["variables"][var]
            out = None
            if var == "sin_lat":
                out = compute_sincos_coords(
                    self.lat, type="sin", samples=self.num_samples
                )
                constant_vars.add(var)
            elif var == "cos_lat":
                out = compute_sincos_coords(
                    self.lat, type="cos", samples=self.num_samples
                )
                constant_vars.add(var)
            elif var == "sin_lon":
                out = compute_sincos_coords(
                    self.lon, type="sin", samples=self.num_samples
                )
                constant_vars.add(var)
            elif var == "cos_lon":
                out = compute_sincos_coords(
                    self.lon, type="cos", samples=self.num_samples
                )
                constant_vars.add(var)
            elif var == "sin_julian_day":
                out = compute_julian_day(
                    dates=pd.to_datetime(self.dates),
                    type="sin",
                    points=self.number_gridpoints,
                )
            elif var == "cos_julian_day":
                out = compute_julian_day(
                    dates=pd.to_datetime(self.dates),
                    type="cos",
                    points=self.number_gridpoints,
                )
            elif var == "toa_solar_radiation":
                out = compute_toa_solar_radiation(
                    dates=pd.to_datetime(self.dates), lats=self.lat
                )
            if out is not None:
                data_arr[:, idx_var, :] = out[:, 0, :]
                log.info("Forcing %s ready.", var)

        # --- variables_metadata and constant_fields ---
        variables_metadata = {}
        for var in self.variables:
            if var in _KNOWN_COMPUTED_FORCINGS:
                variables_metadata[var] = _KNOWN_COMPUTED_FORCINGS[var].copy()
            elif var in constant_vars:
                variables_metadata[var] = {
                    "computed_forcing": False,
                    "constant_in_time": True,
                }
            else:
                variables_metadata[var] = {
                    "computed_forcing": False,
                    "constant_in_time": False,
                }

        zarr_group.attrs["variables_metadata"] = variables_metadata
        zarr_group.attrs["constant_fields"] = sorted(
            v for v, m in variables_metadata.items() if m["constant_in_time"]
        )

        # --- Count NaNs ---
        log.info("Counting NaNs.")
        idx_fixed_nan, idx_dynamic_nan = self.count_nans(zarr_group)
        zarr_group.attrs["idx_fixed_nan"] = {
            var: idx_fixed_nan[c].tolist() for c, var in enumerate(self.variables)
        }
        zarr_group.attrs["idx_dynamic_nan"] = {
            var: idx_dynamic_nan[c] for c, var in enumerate(self.variables)
        }

        # --- Impute NaNs (optional) ---
        if self.imputer is not None:
            zarr_group.attrs["idx_dynamic_nan"] = self.impute_nans(
                zarr_group=zarr_group,
                zarr_attrs=zarr_group.attrs["idx_dynamic_nan"],
                lats=zarr_group["latitudes"][:],
                lons=zarr_group["longitudes"][:],
            )

        # --- Compute and store stats as (C,) sub-arrays (single pass) ---
        log.info("Computing stats.")
        mean_arr, std_arr, min_arr, max_arr = self.compute_stats_per_channel(zarr_group)

        zarr_group.create_dataset(
            "mean",
            data=mean_arr,
            chunks=(self.num_vars,),
            dtype="float32",
            compressor=blk,
        )
        zarr_group.create_dataset(
            "std",
            data=std_arr,
            chunks=(self.num_vars,),
            dtype="float32",
            compressor=blk,
        )
        zarr_group.create_dataset(
            "min",
            data=min_arr,
            chunks=(self.num_vars,),
            dtype="float32",
            compressor=blk,
        )
        zarr_group.create_dataset(
            "max",
            data=max_arr,
            chunks=(self.num_vars,),
            dtype="float32",
            compressor=blk,
        )

        log.info("Saved store at %s", zarr_path)
        return zarr_path
