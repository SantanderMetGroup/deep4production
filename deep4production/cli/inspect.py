import sys
import yaml
from deep4production.utils.zarr import zarr_inspect 

def main():
    """
    Main entry point for the D4P inspect console script.
    Purpose: Inspects a Zarr dataset using the zarr_inspect utility.
    Parameters:
        None (reads sys.argv for zarr file path)
    Returns:
        None
    """
    zarr_path = sys.argv[1]
    zarr_inspect(zarr_path)
