import os
import glob
import numcodecs
import numpy as np
import pandas as pd
import xarray as xr 
import zarr
import yaml

from torch.utils.data import Dataset
from deep4downscaling.utils.trans import xarray_to_numpy
from deep4downscaling.utils.general import is_grid_regular
from deep4downscaling.utils.imputers import d4dimputers
########################################################################################################
class d4d_dataset(Dataset):
  def __init__(self, date_init, date_end, freq, data):
    
    # --- GENERAL INFO ------
    # Self parameters
    self.data = data

    # Number of variables
    self.variables = data["vars"] 
    self.num_vars = len(self.variables)

    # Get sources of netcdf files
    raw_sources = self.data["paths"]
    self.source_files = []
    for source in raw_sources:
        if "*" in source:
            matched_files = sorted(glob.glob(source))
            if not matched_files:
                print(f"⚠️ Warning: No files matched pattern: {source}")
            self.source_files.extend(matched_files)
        elif os.path.isfile(source):
            self.source_files.append(source)
        else:
            print(f"⚠️ Warning: File not found: {source}")
    self.num_sources = len(self.source_files)
    print(f"✅ Found {self.num_sources} source NetCDF files.")

    # --- TEMPORAL INFORMATION ------
    # Number of samples in the YAML
    self.date_init = pd.to_datetime(date_init)
    self.date_end = pd.to_datetime(date_end)
    self.freq = freq
    dates_yaml = pd.date_range(start=self.date_init, end=self.date_end, freq=self.freq).to_numpy()
    self.num_samples_yaml = len(dates_yaml)
    # Number of samples in the source NETCDF files
    available_dates = self.get_available_dates_in_sources(self.source_files)
    self.dates = np.array([d for d in dates_yaml if d in available_dates])
    self.num_samples = len(self.dates)
    # Print info
    emoji = "🟢" if self.num_samples == self.num_samples_yaml else "⚠️"
    print(f"{emoji} Found {self.num_samples} of {self.num_samples_yaml} requested samples from the YAML period in the source NetCDF files.")

    # --- SPATIAL INFORMATION ------
    with xr.open_dataset(self.source_files[0]) as temp:
        self.spatial_dims, self.is_regular = self.get_spatial_dims(temp)
        print(temp)
        if self.is_regular:
            # Store dimensions
            self.H = len(temp[self.spatial_dims[0]].values)
            self.W = len(temp[self.spatial_dims[1]].values)
            # Flatten the spatial grid
            temp = temp.stack(point=self.spatial_dims)
        # Once mapped to irregular (or originally irregular), get number of gridpoints.
        self.number_gridpoints = len(temp[self.spatial_dims[0]].values)
        # Extract flattened lat/lon
        self.lat = temp.lat.values
        self.lon = temp.lon.values
        # Delete temporal file
        del temp

    # --- IMPUTE NANS (if any) ------
    self.imputer = data.get("imputer", None)

  # ---------------------------------------------------------------
  def get_spatial_dims(self, dataset):
    # Detect grid type and spatial dims
    spatial_dims = ["lat", "lon"]
    if {"x", "y"}.issubset(dataset.dims):
        spatial_dims = ["y", "x"]
    # Check regularity
    grid_type = is_grid_regular(dataset)
    # Change order of spatial dimensions 
    var_name = list(dataset.data_vars.keys())[0]
    # get the coordinates associated with that variable
    var_coords = list(dataset[var_name].dims)[1:]
    return var_coords, grid_type

  # ---------------------------------------------------------------
  def get_available_dates_in_sources(self, paths):
    available_dates = []
    for p in paths:
        try:
            with xr.open_dataset(p) as ds:
                times = ds["time"].values
                # Convert to ISO strings regardless of datetime type
                times_str = [str(t) for t in times]
                available_dates.append(times_str)
        except Exception as e:
            print(f"Warning: Could not read {p}: {e}")
    return np.array( np.concat(available_dates), dtype='datetime64[ns]')

  # ---------------------------------------------------------------
  def compute_mean_std_per_channel(self, zarr_path):
    z = zarr.open(zarr_path, mode='r')
    C = z.shape[1]
    count = np.zeros(C, dtype=np.int64)
    sum_ = np.zeros(C, dtype=np.float64)
    delta_squared = np.zeros(C, dtype=np.float64)
    # -- Mean --
    for i in range(z.shape[0]):
        # Sample
        x = z[i].astype(np.float64) # Shape (C, H*W)
        # Count NaNs
        nan_mask = np.isnan(x)
        # Mean and count
        sum_ += np.nansum(x, axis=1)
        count += np.sum(~nan_mask, axis=1)
    mean = sum_ / count
    # -- Std --
    for i in range(z.shape[0]):
        # Sample
        x = z[i].astype(np.float64) # Shape (C, H*W)
        # Std
        delta_squared += np.nansum( (x - mean[:, None]) ** 2, axis=1)
    std = np.sqrt(delta_squared / (count - 1))
    # Return
    m_dict = { var: float(m) for var, m in zip(self.variables, mean.astype(np.float32)) }
    std_dict = { var: float(m) for var, m in zip(self.variables, std.astype(np.float32)) }
    return m_dict, std_dict

  # ---------------------------------------------------------------
  def compute_min_max_per_channel(self, zarr_path):
    z = zarr.open(zarr_path, mode='r')
    C = z.shape[1]  # number of channels (variables)
    min_vals = np.full(C, np.inf, dtype=np.float32)
    max_vals = np.full(C, -np.inf, dtype=np.float32)
    for i in range(z.shape[0]):  # over time steps
        x = z[i].astype(np.float32)  # Shape (C, H*W)
        min_vals = np.minimum(min_vals, np.nanmin(x, axis=1))
        max_vals = np.maximum(max_vals, np.nanmax(x, axis=1))
    min_dict = { var: float(m) for var, m in zip(self.variables, min_vals.astype(np.float32)) }
    max_dict = { var: float(m) for var, m in zip(self.variables, max_vals.astype(np.float32)) }
    return min_dict, max_dict

  # ---------------------------------------------------------------
  def count_nans(self, zarr_path):
    z = zarr.open(zarr_path, mode='r')[:]
    S, C, G = z.shape
    z = z.astype(np.float64) # (S, C, G=H*W)
    nan_mask = np.isnan(z)
    nan_count = np.sum(nan_mask, axis=0)  # Count NaNs per (variable, grid point). shape: (C, G)
    nan_fixed_dict = {}
    nan_dynamic_dict = {}
    for v, var in enumerate(self.variables):
        # --- Fixed NaNs: always NaN across all samples ---
        idx_fixed_nan = np.where(nan_count[v] == S)[0]
        nan_fixed_dict[var] = None if len(idx_fixed_nan) == 0 else idx_fixed_nan
        # --- Dynamic NaNs: NaN sometimes, valid sometimes ---
        dynamic_nan_list = []
        for s in range(self.num_samples):
            idxs_dynamic_nan = np.where(nan_mask[s,v,:])[0] # Includes both fixed and dynamic
            idxs_dynamic_nan = [
                i for i in idxs_dynamic_nan 
                if i not in set(nan_fixed_dict.get(var, []))
            ] # Subset only dynamic NaNs
            if len(idxs_dynamic_nan) > 0: 
                for idx_dynamic_nan in idxs_dynamic_nan:
                    dynamic_nan_list.append([s, idx_dynamic_nan])
        nan_dynamic_dict[var] = None if len(dynamic_nan_list) == 0 else dynamic_nan_list
        # --- Print ---
        # print(f"{var}: fixed NaNs={len(idx_fixed_nan)}, dynamic NaNs={len(idx_dynamic_nan)}")
    # --- Return ---
    return nan_fixed_dict, nan_dynamic_dict

  # ---------------------------------------------------------------
  def impute_nans(self, data, zarr_attrs, lats, lons):
    for var in self.variables:
        # Get idx of variable in zarr object
        idx_var = data.attrs['variables'][var] 
        # Get imputer
        imputer_default = self.imputer.get("default", None)
        imputer_selected = self.imputer.get(var, imputer_default)
        imputer_func_str = imputer_selected["name"]
        if self.idx_dynamic_nan[var] is not None:
            # Get kwargs
            kwargs_imputer = {k: v for k, v in imputer_selected.items() if k != "name"}
            num_dates_nan_dynamic = len(self.idx_dynamic_nan[var])
            for s in range(num_dates_nan_dynamic):
                idx_t = self.idx_dynamic_nan[var][s][0]
                idx_gp = self.idx_dynamic_nan[var][s][1]
                print(f"🔧 [{var}] Using imputer: {imputer_func_str} to replace date: {idx_t}:{self.dates[idx_t]} at gridpoint: {idx_gp}")
                imp = d4dimputers(data=data[s,idx_var,:], lat_gp=lats[idx_gp], lon_gp=lons[idx_gp], lats_ref=lats, lons_ref=lons)
                imputer_func = getattr(imp, imputer_func_str)
                data[idx_t,idx_var,idx_gp] = imputer_func(**kwargs_imputer)   
            zarr_attrs[var] = []
        else:
            print(f"⚠️  [{var}] Imputer {imputer_func_str} selected, but there are no dynamic NaNs in the dataset. Skipping.")
    # Return
    return data, zarr_attrs

  # ---------------------------------------------------------------
  def to_disk(self, zarr_path):
    # --- Initialize zarr ---------------------------------
    zarr_store = zarr.open(
        zarr_path,
        mode='w',
        shape=(self.num_samples, self.num_vars, self.number_gridpoints),
        chunks=(1, self.num_vars, self.number_gridpoints),
        dtype="float32",
        compressor=numcodecs.Blosc(cname='zstd', clevel=5),
        zarr_format=2,
        fill_value=np.nan
    )

    # --- Attributes ---------------------------------
    zarr_store.attrs['dates'] = [str(date) for date in self.dates]
    zarr_store.attrs['date_init_yaml'] = str(self.date_init)
    zarr_store.attrs['date_end_yaml'] = str(self.date_end)
    zarr_store.attrs['num_samples'] = self.num_samples
    zarr_store.attrs['num_samples_yaml'] = self.num_samples_yaml
    zarr_store.attrs['temporal_freq'] = self.freq
    zarr_store.attrs['variables'] = {var: idx  for idx, var in enumerate(self.variables)}
    zarr_store.attrs['name_dims'] = ["time", "variable", "gridpoint"] 
    zarr_store.attrs['shape'] = [len(self.dates), self.num_vars, self.number_gridpoints]
    zarr_store.attrs['lats'] = [lat for lat in self.lat]
    zarr_store.attrs['lons'] = [lon for lon in self.lon]
    zarr_store.attrs['is_regular'] = self.is_regular
    if self.is_regular:
        zarr_store.attrs['H'], zarr_store.attrs['W'] = self.H, self.W

    # --- Data ---------------------------------
    sources = self.source_files
    for source in sources:
        x = xr.open_dataset(source)      
        vars_source = x.data_vars

        for var in vars_source:
            if var in self.variables:
                print(f"✅ Variable {var} from {source} matches target variables.")
                idx_var = zarr_store.attrs['variables'][var]

                # Load data
                x_ = x[[var]]

                # Temporal intersection
                avail_dates_in_source = x_.time.values.astype('datetime64[ns]')
                matching_dates = np.intersect1d(self.dates, avail_dates_in_source)
                idx_samples = [np.where(self.dates == t)[0][0] for t in matching_dates]
                x_ = x_.sel(time=matching_dates)
                
                # Flatten spatial dimension
                if self.is_regular:
                    x_ = x_.stack(point=self.spatial_dims)

                # From xarray to numpy
                xdata = xarray_to_numpy(x_).astype(np.float32)
                del x_

                # Write data block
                for i, t_idx in enumerate(idx_samples):
                    zarr_store[t_idx, idx_var, :] = xdata[i]
            else:
                print(f"⚠️ Skipping variable {var} in {source} not in target variable list.")
        
        x.close()        
        del x


    # Count nans 
    idx_fixed_nan, self.idx_dynamic_nan = self.count_nans(zarr_path)
    zarr_store.attrs['idx_fixed_nan'] = {
        var: [] if v is None else [int(n) for n in v] for var, v in idx_fixed_nan.items()
    }
    # print(zarr_store.attrs['idx_fixed_nan'])
    zarr_store.attrs['idx_dynamic_nan'] = {
       var: [] if v is None else [[int(s), int(g)] for s, g in v]
       for var, v in self.idx_dynamic_nan.items()
    }
    # print(zarr_store.attrs['idx_dynamic_nan'])

    # Impute NaNs?
    if self.imputer is not None:
        zarr_store, zarr_store.attrs['idx_dynamic_nan'] = self.impute_nans(data=zarr_store, zarr_attrs=zarr_store.attrs['idx_dynamic_nan'], lats=zarr_store.attrs['lats'], lons=zarr_store.attrs['lons'])

    # Compute mean/std, min/max
    m, s = self.compute_mean_std_per_channel(zarr_path)
    zarr_store.attrs['mean'] = m
    zarr_store.attrs['std'] = s
    mn, mx = self.compute_min_max_per_channel(zarr_path)
    zarr_store.attrs['min'] = mn
    zarr_store.attrs['max'] = mx

    # Save to disk
    return f"Saved to disk...: {zarr_path}"

