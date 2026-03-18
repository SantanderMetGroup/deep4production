import yaml
import numpy as np
import xarray as xr
import pandas as pd
import importlib

# --------------------------------------------------------------------------------------------------------------
def get_func_from_string(module_string, func_string, kwargs=None):
    """
    Dynamically imports and returns a function or class from a module.
    Parameters:
        module_string (str): Module path as string.
        func_string (str): Function or class name as string.
        kwargs (dict, optional): Keyword arguments for instantiation.
    Returns:
        Callable or object: Imported function or class instance.
    """
    module = importlib.import_module(module_string)
    func = getattr(module, func_string)
    return func(**kwargs) if kwargs is not None else func

# --------------------------------------------------------------------------------------------------------------
def read_metadata_from_yaml(yaml_path: str) -> dict:
    with open(f"{yaml_path}", "r") as f:
        metadata = yaml.safe_load(f)
    return metadata

# --------------------------------------------------------------------------------------------------------------
def is_grid_regular(ds: xr.Dataset) -> bool:
    """
    Determine if a dataset is defined on a regular grid, based on
    dimensionality.

    Logic:
    - Identify the main data variable(s).
    - Check which dimensions correspond to coordinate axes.
    - If a data variable has ≥2 spatial dimensions (e.g. lat/lon or y/x)
      that are coordinates of the dataset, it is considered regular.
    - Otherwise, it is irregular (station, unstructured, etc.)
    """
    
    # Identify likely spatial dimensions
    dims = [d for d in ds.dims if d in ("lat", "lon", "x", "y")]
    spatial_dims = [d for d in ds.dims if d in ("lat", "lon", "x", "y")]
    if not spatial_dims:
        return False
    # Find a representative data variable (skip scalars or coords)
    data_vars = [v for v in ds.data_vars if ds[v].ndim > 0]
    if not data_vars:
        return False
    for var in data_vars:
        dims = ds[var].dims
        # Count how many of its dims are spatial ones
        spatial_count = sum(d in spatial_dims for d in dims)
        if spatial_count >= 2:
            return True  # Regular gridded (2D+ spatial structure)
    return False  # Likely station-based or 1D

def latlon_to_xyz(lat, lon):
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    x = np.cos(lat)*np.cos(lon)
    y = np.cos(lat)*np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)