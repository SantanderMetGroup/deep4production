# Image placeholders

The tutorial markdown references the following images. Run training / inference to capture them, or copy reusable ones from the DeepESD tutorial.

| Image | What it shows | How to capture |
|---|---|---|
| `d4p-train-output.png` | Console output of `d4p-train ./training/configs/cpmgem.yaml` (epoch logs, loss curves) | Screenshot the terminal partway through training |
| `d4p-downscale-output.png` | Console output of `d4p-downscale ./inference/configs/cpmgem.yaml` (sampling progress) | Screenshot the terminal during inference |
| `d4p-downscale-pred.png` | `xarray` preview of the resulting `1980.nc` showing `(member, time, point)` dims | `xr.open_dataset("./outputs/cpmgem/predictions/1980.nc")` in a notebook, then screenshot the repr |
| `cpmgem_1980-01-01.png` | Output of the `plot_date_from_1D_spatial_field` snippet in section 8 | Save the matplotlib figure produced by the snippet |
