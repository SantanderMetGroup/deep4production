import yaml
import numpy as np
import xarray as xr
import pandas as pd
import importlib
# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr

# --------------------------------------------------------------------------------------------------------------
# def c4r2d4d(c4r):

#     """
#     This function transforms a .RData object returned by climate4R functions into an
#     xarray.Dataset for further manipulation and analysis in Python. It extracts key
#     components from the input R object, such as the variable(s), data, spatial coordinates,
#     temporal information, and optional metadata, and reorganizes them into an xarray.Dataset.
#     The function maps spatial coordinates to longitude (lon) and latitude (lat) for gridded
#     data or assigns them to a loc dimension for flattened datasets. Temporal information is
#     converted into Python's datetime format for easier time-based indexing. The function
#     dynamically handles multiple dimensions, such as time, lat, lon, loc, and member,
#     depending on the structure of the input data. It ensures that the resulting dataset is
#     compatible with Python's geospatial libraries, allowing users to efficiently work with climate
#     data in Python. Additionally, any metadata from the input R object is preserved and included as
#     global attributes in the output xarray.Dataset.

#     Parameters
#     ----------
#     c4r
#         The .RData object returned by climate4R functions containing the climate data and metadata.
        
#     Returns
#     -------
#     xarray.Dataset 
#         The transformed dataset in Python's xarray format, with labeled dimensions and coordinates for easy manipulation and analysis.
#     """
#     base = importr('base')
    
#     # Extract components 
#     variable = c4r.rx2('Variable')  
#     data = c4r.rx2('Data')  
#     xyCoords = c4r.rx2('xyCoords')  
#     dates = c4r.rx2('Dates')  
#     metadata = c4r.rx2('Metadata') if 'Metadata' in c4r.names else None

#     dim_names = base.attr(data, 'dimensions')  
#     var_names = list(variable[0])  

#     # Convert data to numpy array
#     data_array = np.array(data)  

#     # Extract coordinates
#     lon = np.array(xyCoords.rx2('x'))
#     lat = np.array(xyCoords.rx2('y'))

#     # Extract time information
#     dates_start = np.array(dates.rx2('start') if len(variable.rx2('varName')) == 1 else dates[0].rx2('start'))
#     time = pd.to_datetime(dates_start).tz_localize(None)

#     # Create dynamic Dataset with the variables involved
    
#     data_vars = {}
#     loc_x, loc_y = np.array(xyCoords.rx2('x')), np.array(xyCoords.rx2('y'))

#     def create_dataset(var_name, dims, data_slice, coords):
#         data_vars[var_name] = (dims, data_slice)
#         return xr.Dataset(data_vars=data_vars, coords=coords)

#     if "var" in dim_names:
#         for idx, var_name in enumerate(var_names):
#             if "loc" in dim_names:
#                 if "member" in dim_names:
#                     dims = ["member", "time", "loc"]
#                     if "Members" in c4r.names:
#                         members = seasonal.rx2('Members')
#                         new_ds = create_dataset(
#                             var_name, dims, data_array[ :, :, :],
#                             coords={
#                                 "lon": ("loc", loc_x),
#                                 "lat": ("loc", loc_y),
#                                 "time": ("time", time),
#                                 "member": ("member", members)
#                             }
#                         )
#                     else:
#                         idx_mem = np.where(np.array(dim_names) == 'member')[0][0]
#                         new_ds = create_dataset(
#                             var_name, dims, data_array[idx, :, :, :],
#                             coords={
#                                 "lon": ("loc", loc_x),
#                                 "lat": ("loc", loc_y),
#                                 "time": ("time", time),
#                                 "member": ("member", np.arange(data_array.shape[idx_mem]))
#                             }
#                         )
#                 else:
#                     dims = ["time", "loc"]
#                     new_ds = create_dataset(
#                         var_name, dims, data_array[idx, :, :],
#                         coords={
#                             "lon": ("loc", loc_x),
#                             "lat": ("loc", loc_y),
#                             "time": ("time", time)
#                         }
#                     )
#             else:
#                 if "member" in dim_names:
#                     dims = ["member", "time", "lat", "lon"]
#                     if "Members" in c4r.names:
#                         members = seasonal.rx2('Members')
#                         new_ds = create_dataset(
#                         var_name, dims, data_array[ :, :, :, :],
#                         coords={
#                             "lon": ("lon", lon),
#                             "lat": ("lat", lat),
#                             "time": ("time", time),
#                             "member": ("member", members)
#                         }
#                     )
#                     else: 
#                         idx_mem = np.where(np.array(dim_names) == 'member')[0][0]
#                         new_ds = create_dataset(
#                             var_name, dims, data_array[idx, :, :, :, :],
#                             coords={
#                                 "lon": ("lon", lon),
#                                 "lat": ("lat", lat),
#                                 "time": ("time", time),
#                                 "member": ("member", np.arange(data_array.shape[idx_mem]))
#                             }
#                         )
#                 else:
#                     dims = ["time", "lat", "lon"]
#                     new_ds = create_dataset(
#                         var_name, dims, data_array[idx, :, :, :],
#                         coords={
#                             "lon": ("lon", lon),
#                             "lat": ("lat", lat),
#                             "time": ("time", time)
#                         }
#                     )
#     else:
#         if "loc" in dim_names:
#             if "member" in dim_names:
#                 dims = ["member", "time", "loc"]
#                 if "Members" in c4r.names:
#                     members = seasonal.rx2('Members')
#                     new_ds = create_dataset(
#                         var_name, dims, data_array[ :, :, :, :],
#                         coords={
#                             "lon": ("loc", loc_x),
#                             "lat": ("loc", loc_y),
#                             "time": ("time", time),
#                             "member": ("member", members)
#                         }
#                     )
#                 else:
#                     idx_mem = np.where(np.array(dim_names) == 'member')[0][0]
#                     new_ds = create_dataset(
#                         var_names[0], dims, data_array[:, :, :],
#                         coords={
#                             "lon": ("loc", loc_x),
#                             "lat": ("loc", loc_y),
#                             "time": ("time", time),
#                             "member": ("member", np.arange(data_array.shape[idx_mem]))
#                         }
#                     )
#             else:
#                 dims = ["time", "loc"]
#                 new_ds = create_dataset(
#                     var_names[0], dims, data_array[:, :],
#                     coords={
#                         "lon": ("loc", loc_x),
#                         "lat": ("loc", loc_y),
#                         "time": ("time", time)
#                     }
#                 )
#         else:
#             if "member" in dim_names:
#                 dims = ["member", "time", "lat", "lon"]
#                 if "Members" in c4r.names:
#                     members = seasonal.rx2('Members')
#                     new_ds = create_dataset(
#                         var_names[0], dims, data_array[:, :, :, :],
#                         coords={
#                             "lon": ("lon", lon),
#                             "lat": ("lat", lat),
#                             "time": ("time", time),
#                             "member": ("member", members)
#                         }
#                     )
#                 else:
#                     idx = np.where(np.array(dim_names) == 'member')[0][0]
#                     new_ds = create_dataset(
#                         var_names[0], dims, data_array[:, :, :, :],
#                         coords={
#                             "lon": ("lon", lon),
#                             "lat": ("lat", lat),
#                             "time": ("time", time),
#                             "member": ("member", np.arange(data_array.shape[idx_mem]))
#                         }
#                     )
#             else:
#                 dims = ["time", "lat", "lon"]
#                 new_ds = create_dataset(
#                     var_names[0], dims, data_array[:, :, :],
#                     coords={
#                         "lon": ("lon", lon),
#                         "lat": ("lat", lat),
#                         "time": ("time", time)
#                     }
#                 )             
#     # Assign metadata as attributtes if present
#     if metadata is not None:
#         new_ds.attrs = dict(zip(list(metadata.names), list(metadata)))
#         # Assign metadata as attributes
#         new_ds.attrs = dict(zip(list(metadata.names), list(metadata)))
#     return new_ds



# --------------------------------------------------------------------------------------------------------------
def get_func_from_string(module_string: str, func_string: str, kwargs: dict=None):
    """Dynamically import a function or class from module path."""
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