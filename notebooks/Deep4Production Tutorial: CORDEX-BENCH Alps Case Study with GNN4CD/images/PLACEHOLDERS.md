# Image placeholders

The tutorial markdown references the following images. Run training / inference to capture them, or copy reusable ones from the DeepESD tutorial.

| Image | What it shows | How to capture |
|---|---|---|
| `d4p-train-output.png` | Console output of `d4p-train ./training/configs/gnn4cd_asym.yaml` — should include the "Building graph" stage on first run | Screenshot the terminal during training |
| `d4p-downscale-output.png` | Console output of `d4p-downscale ./inference/configs/gnn4cd_asym.yaml` | Screenshot the terminal during inference |
| `d4p-downscale-pred.png` | `xarray` preview of the resulting `1980.nc` (predictions are in physical units thanks to the inverse `log1p`) | `xr.open_dataset("./outputs/gnn4cd_asym/predictions/1980.nc")`, screenshot the repr |
| `gnn4cd_1980-01-01.png` | Output of the `plot_date_from_1D_spatial_field` snippet in section 8 | Save the matplotlib figure produced by the snippet |
