# Image placeholders

The tutorial markdown references the following images. Run training / inference to capture them, or copy reusable ones from the DeepESD tutorial.

| Image | What it shows | How to capture |
|---|---|---|
| `d4p-train-output-regressor.png` | Console output of `d4p-train ./training/configs/song_unet_det.yaml` (step 6.1) | Screenshot the terminal during regressor training |
| `d4p-train-output-resdiff.png` | Console output of `d4p-train ./training/configs/resdiff.yaml` (step 6.2). Should include the "Producing residuals (netcdf and zarr files)" stage on first run | Screenshot the terminal during diffusion training |
| `d4p-downscale-output.png` | Console output of `d4p-downscale ./inference/configs/resdiff.yaml` | Screenshot the terminal during inference |
| `d4p-downscale-pred.png` | `xarray` preview of the resulting `1980.nc` with `(member, time, point)` dims | `xr.open_dataset("./outputs/resdiff/predictions/1980.nc")` in a notebook, then screenshot the repr |
| `resdiff_1980-01-01.png` | Output of the `plot_date_from_1D_spatial_field` snippet in section 8 | Save the matplotlib figure produced by the snippet |
