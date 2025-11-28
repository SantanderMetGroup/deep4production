## Load libraries
import zarr
import numpy as np 

##################################################################################################################################
def zarr_inspect(
    zarr_path: str,
):

    # Load the Zarr store
    zarr_store = zarr.open(zarr_path, mode='r')
    
    # General Info
    # date_init = zarr_store.attrs.get('dates', [None])[0]
    # date_end = zarr_store.attrs.get('dates', [None])[-1]
    date_init = zarr_store.attrs.get('date_init_yaml')
    date_end = zarr_store.attrs.get('date_end_yaml')
    freq = zarr_store.attrs.get('freq', [None])[0]
    num_samples = zarr_store.attrs.get('num_samples', 'Unknown')
    num_samples_yaml = zarr_store.attrs.get('num_samples_yaml', 'Unknown')
    
    coords_lats = zarr_store.attrs.get('lats', [])
    coords_lons = zarr_store.attrs.get('lons', [])
    is_regular = zarr_store.attrs.get('is_regular', [])
    name_dims = zarr_store.attrs.get('name_dims', [])
    shape = zarr_store.attrs.get('shape', [])

    variables = zarr_store.attrs.get('variables', [])
    num_variables = len(variables)
    means = zarr_store.attrs.get('mean', [])
    stds = zarr_store.attrs.get('std', [])
    mins = zarr_store.attrs.get('min', [])
    maxs = zarr_store.attrs.get('max', [])

    idx_fixed_nan = zarr_store.attrs.get('idx_fixed_nan', [])
    idx_dynamic_nan = zarr_store.attrs.get('idx_dynamic_nan', [])

    # PRINT ------------------------------------------------------
    print("-" * 170)
    print("General Information 📈🤖📊")
    print("-" * 170)
    print(f"{'Date Init (requested at creation):'} {date_init}")
    print(f"{'Date End (requested at creation):'} {date_end}")
    print(f"Number of samples available: {num_samples}/{num_samples_yaml}")

    print()

    print(f"Latitude range: {np.min(coords_lats)} to {np.max(coords_lats)} degrees")
    print(f"Longitude range: {np.min(coords_lons)} to {np.max(coords_lons)} degrees")
    print(f"Is the data on a regular grid?: {is_regular}")

    print()

    print(f"Name of dimensions: {name_dims}")
    print(f"Dimensions: {shape}")
    
    print()
    print()

    print("-" * 170)
    print("Variables Summary 📊📉📈")
    print("-" * 170)
    print(f"{'Variable':22} | {'Mean':>10} | {'Std':>10} | {'Min':>10} | {'Max':>10} | {'Fixed NaNs (Number of gridpoints)':>10} | {'Dynamic NaNs (Number of samples)':>10}")
    print("-" * 170)
    
    for var, idx in variables.items():
        m = f"{means[var]:.4f}" 
        s = f"{stds[var]:.4f}" 
        mn = f"{mins[var]:.4f}" 
        mx = f"{maxs[var]:.4f}" 
        nf = f"{len(idx_fixed_nan[var]):.0f}" 
        nd = f"{len(idx_dynamic_nan[var]):.0f}" 
        print(f"{var:22} | {m:>10} | {s:>10} | {mn:>10} | {mx:>10} | {nf:>33} | {nd:>33} ")
    
    print("-" * 170)
    


