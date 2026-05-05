import sys
import argparse
from deep4production.utils.zarr import zarr_inspect


def main():
    parser = argparse.ArgumentParser(
        prog="d4p-inspect",
        description="Inspect a d4p or anemoi-datasets zarr store.",
    )
    parser.add_argument("zarr_path", help="Path to the zarr store.")
    parser.add_argument(
        "--format", "-f",
        choices=["auto", "d4p", "anemoi"],
        default="auto",
        dest="fmt",
        help="Force zarr format (default: auto-detect).",
    )
    args = parser.parse_args()
    zarr_inspect(args.zarr_path, fmt=args.fmt)
