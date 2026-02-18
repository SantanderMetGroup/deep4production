import sys
import yaml
from deep4production.utils.zarr import zarr_inspect 

def main():
    zarr_path = sys.argv[1]
    zarr_inspect(zarr_path)
