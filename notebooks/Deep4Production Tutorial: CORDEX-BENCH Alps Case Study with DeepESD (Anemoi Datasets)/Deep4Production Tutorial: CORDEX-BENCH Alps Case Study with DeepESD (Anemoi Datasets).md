# Deep4Production Tutorial: CORDEX-BENCH Alps Case Study with DeepESD (Anemoi Datasets)

This tutorial demonstrates how to use **anemoi-datasets** as the data-ingest backend for the [deep4production](https://github.com/SantanderMetGroup/deep4production) framework. It covers the same CORDEX-BENCH Alps / DeepESD case study as the companion notebook, but replaces `d4p-create` with `anemoi-datasets create` to produce the AI-ready Zarr stores. All downstream steps (`d4p-inspect`, `d4p-train`, `d4p-downscale`) are unchanged — a single `format: anemoi` key in the YAML configuration is all that is needed to tell deep4production to read anemoi Zarr stores.

______________________________________________________________________

## 1. Introduction

### When to use anemoi-datasets instead of d4p-create

[anemoi-datasets](https://github.com/ecmwf/anemoi-datasets) is ECMWF's open-source library for building AI-ready datasets for weather and climate applications. It natively supports a wide variety of data sources (GRIB, NetCDF, MARS, OpenDAP, Zarr, …), applies CF-convention coordinate auto-detection, handles multi-source joins, and computes statistics in a single pass.

Use `anemoi-datasets create` instead of `d4p-create` when:

- Your team already produces anemoi Zarr stores for other workflows (e.g., [anemoi-training](https://github.com/ecmwf/anemoi-training))
- You need to ingest data from MARS, GRIB, or remote object stores that `d4p-create` does not support
- You want to leverage anemoi's built-in source ecosystem (accumulations, recentering, multi-member ensembles, …)

Use `d4p-create` when:

- You are working exclusively with local NetCDF files and do not need anemoi's extended source ecosystem
- You need to ingest static (time-independent) fields such as orography — `d4p-create` handles time-free variables natively
- You prefer a simpler, single-tool pipeline

> **Note on static fields.** `anemoi-datasets` is designed for time-varying data and does not natively handle time-independent (static) fields such as orography. For this reason the `forcings` / `orog` block is omitted in this tutorial. If you need static fields, use `d4p-create` for those Zarr stores and mix `format: d4p` with `format: anemoi` in the training YAML (see Section 6).

### deep4production integration

deep4production reads anemoi Zarr stores transparently through a thin read-only adapter (`AnemoiZarrStore`). The adapter translates anemoi's internal layout — `data` array of shape `(T, V, E, G)`, stats named `stdev/minimum/maximum`, variables indexed via `name_to_index` — into the d4p v2 contract used by pydatasets, downscalers, and the normalizer. The only change required in deep4production YAML files is adding:

```yaml
format: anemoi
```

to each predictor/predictand/input_data block that points to an anemoi Zarr store.

### Command Line Interface (CLI)

This tutorial uses the following CLIs:

| Tool | Purpose |
|------|---------|
| `anemoi-datasets create` | Build AI-ready Zarr from NetCDF/GRIB/remote sources |
| `d4p-inspect` | Quick QA/QC of any Zarr (d4p or anemoi format) |
| `d4p-train` | Train a deep learning downscaling model |
| `d4p-downscale` | Run inference with a trained model |

______________________________________________________________________

## 2. Case Study: CORDEX-BENCH Alps

This tutorial uses the same simplified CORDEX-BENCH configuration as the companion DeepESD notebook. Please refer to that notebook for a full description of the case study. In summary:

- **Domain:** Central Europe (Alps)
- **Model:** DeepESD with Bernoulli-Gamma loss
- **Predictors:** 15 upper-air variables from the upscaled CNRM-CM5-ALADIN-63 RCM at 2° resolution (16 × 16)
- **Predictands:** Daily precipitation (`pr`) from CNRM-CM5-ALADIN-63 at 0.11° resolution (128 × 128)
- **Training period:** 1961–1980 (excluding 1967, 1975 held out for validation)
- **Test period:** 1980

______________________________________________________________________

## 3. Download CORDEX-BENCH Alps Data

The data download step is identical to the companion notebook. Place the data in `./source_files/data_zenodo/`.

```python
import os
import zipfile
import shutil

os.makedirs("./source_files/data_zenodo/", exist_ok=True)

!wget -P ./source_files/data_zenodo/ https://zenodo.org/records/15797226/files/ALPS_domain.zip?download=1

shutil.move("./source_files/data_zenodo/ALPS_domain.zip?download=1",
            "./source_files/data_zenodo/ALPS_domain.zip")

with zipfile.ZipFile('./source_files/data_zenodo/ALPS_domain.zip', 'r') as zip_ref:
    zip_ref.extractall('./source_files/data_zenodo/')

os.remove("./source_files/data_zenodo/ALPS_domain.zip")
```

______________________________________________________________________

## 4. Build AI-Ready Zarr Stores with `anemoi-datasets create`

### How anemoi-datasets works

`anemoi-datasets create` reads a **recipe YAML** that describes:

- The **date range** and **frequency** of the dataset
- The **input sources** (NetCDF, GRIB, MARS, …) and how to map their coordinates
- The **statistics** time window (for mean, std, min, max)

It then writes a self-contained Zarr store with the following internal layout (anemoi format):

```
output.zarr/
├── data          # float32 array (T, V, E, G) — time × variable × ensemble × grid
├── dates         # datetime64[s] array (T,)
├── latitudes     # float64 array (G,) — flattened grid latitudes
├── longitudes    # float64 array (G,) — flattened grid longitudes
├── mean          # float32 array (V,) — per-variable mean
├── stdev         # float32 array (V,) — per-variable standard deviation
├── minimum       # float32 array (V,) — per-variable minimum
└── maximum       # float32 array (V,) — per-variable maximum
```

deep4production's `AnemoiZarrStore` adapter transparently maps `stdev/minimum/maximum` to the `std/min/max` names used internally by d4p, and squeezes the ensemble axis `E`.

### 4.1 Predictors: UPSRCM (15 upper-air variables, 16 × 16)

The predictor NetCDF file has a **regular lat/lon grid** with CF-standard coordinate names (`lat`, `lon`, `time`). anemoi-datasets auto-detects these via their `standard_name` attributes — no explicit `flavour` is needed.

```yaml
# File: ./AI_ready_datasets/configs/UPSRCM_1961-1980_anemoi.yaml

description: CORDEX-BENCH CNRM-CM5 upscaled RCM predictors — 15 upper-air variables,
  2-degree regular grid (16x16), daily, 1961–1980

name: UPSRCM_1961-1980

dates:
  start: 1961-01-01T12:00:00
  end: 1980-12-31T12:00:00
  frequency: 24h

input:
  netcdf:
    path: ./source_files/data_zenodo/ALPS_domain/train/ESD_pseudo_reality/predictors/CNRM-CM5_1961-1980.nc

statistics:
  start: 1961-01-01
  end: 1980-12-31
```

Build the predictor Zarr:

```bash
anemoi-datasets create \
    ./AI_ready_datasets/configs/UPSRCM_1961-1980_anemoi.yaml \
    ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr \
    --overwrite
```

> **Coordinate auto-detection.** The predictor file carries CF-standard attributes on all coordinates (`lat`: `standard_name=latitude`, `lon`: `standard_name=longitude`, `time`: `axis=T`). anemoi-datasets detects a **regular lat/lon (meshed) grid** and produces a flattened output of G = 16 × 16 = 256 grid points.

### 4.2 Predictands: RCM precipitation (curvilinear 128 × 128 grid)

The target NetCDF file uses a **curvilinear (rotated) grid**. The spatial dimensional coordinates are `y` (projection northing, km) and `x` (projection easting, km), while `lat` and `lon` are 2D auxiliary coordinate arrays of shape `(y, x)` carrying the true geographic positions. Both carry CF-standard attributes:

| Coordinate | `standard_name` | Role in anemoi |
|-----------|-----------------|----------------|
| `time` | `time` | Date axis |
| `y` | `projection_y_coordinate` | Grid y-dimension |
| `x` | `projection_x_coordinate` | Grid x-dimension |
| `lat` `(y, x)` | `latitude` | Geographic latitude of each grid point |
| `lon` `(y, x)` | `longitude` | Geographic longitude of each grid point |

anemoi-datasets' coordinate guesser checks `standard_name` attributes in priority order: longitude and latitude before x and y. It therefore correctly classifies `lat`/`lon` as geographic coordinates and identifies a **2D unstructured grid**, producing G = 128 × 128 = 16 384 grid points in the output Zarr.

```yaml
# File: ./AI_ready_datasets/configs/RCM_1961-1980_anemoi.yaml

description: CORDEX-BENCH CNRM-CM5 RCM target — daily precipitation and maximum
  temperature, curvilinear grid (128x128, ~0.11-degree), 1961–1980

name: RCM_1961-1980

dates:
  start: 1961-01-01T12:00:00
  end: 1980-12-31T12:00:00
  frequency: 24h

input:
  netcdf:
    path: ./source_files/data_zenodo/ALPS_domain/train/ESD_pseudo_reality/target/pr_tasmax_CNRM-CM5_1961-1980.nc

statistics:
  start: 1961-01-01
  end: 1980-12-31
```

Build the predictand Zarr:

```bash
anemoi-datasets create \
    ./AI_ready_datasets/configs/RCM_1961-1980_anemoi.yaml \
    ./AI_ready_datasets/files/RCM_1961-1980.zarr \
    --overwrite
```

> **Note.** This Zarr contains `pr` and `tasmax`. The static orography field (`orog`) lives in a separate time-independent file that `anemoi-datasets` cannot ingest directly. The `forcings` block in the training YAML is therefore commented out in this tutorial (it was already commented out in the companion notebook). If you need `orog` as a forcing, create a separate d4p Zarr with `d4p-create` and reference it with `format: d4p` in the training YAML.

______________________________________________________________________

## 5. Inspect the Zarr Stores with `d4p-inspect`

`d4p-inspect` supports both d4p-native and anemoi Zarr stores. Run it exactly as in the companion notebook:

```bash
d4p-inspect ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr  # Predictors
d4p-inspect ./AI_ready_datasets/files/RCM_1961-1980.zarr     # Predictands
```

The output is identical to the d4p-native format: variable names, spatial dimensions, temporal range, stored statistics, and missing-value counts. This confirms that the anemoi adapter exposes all the information that d4p needs.

______________________________________________________________________

## 6. Train a Model with `d4p-train`

The training configuration is **identical** to the companion DeepESD notebook with two additions: `format: anemoi` is added to each `predictors` and `predictands` block to tell deep4production which zarr adapter to use.

```yaml
# File: ./training/configs/deepesd_anemoi.yaml

##### GENERAL INFO #####
run_ID: deepesd
output_dir: ./outputs
overwrite: true

##### TRAINING DATA CONFIGURATION #####
data:
  load_in_memory: true
  training_period:  [1961, 1962, 1963, 1964, 1965, 1966, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1976, 1977, 1978, 1979, 1980]
  validation_period: [1967, 1975]

  predictors:
    paths:
      - ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr
    format: anemoi   # <-- tells d4p to open this Zarr via AnemoiZarrStore
    variables: [u_850, u_700, u_500, v_850, v_700, v_500, t_850, t_700, t_500, q_850, q_700, q_500, z_850, z_700, z_500]
    normalizer:
      path_reference: ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr
      default: mean_std
      q_850: std
      q_700: std
      q_500: std
    transform_to_2D: True

  predictands:
    paths:
      - ./AI_ready_datasets/files/RCM_1961-1980.zarr
    format: anemoi   # <-- tells d4p to open this Zarr via AnemoiZarrStore
    variables:
      - pr
    normalizer: null
    transform_to_2D: True

  # forcings:          # orog not available as an anemoi Zarr (static field).
  #   variables:       # Create a d4p Zarr with d4p-create and use format: d4p here.
  #     - orog
  #   normalizer:
  #     path_reference: ./AI_ready_datasets/files/RCM_1961-1980.zarr
  #     orog: max

##### DATA LOADER CONFIGURATION #####
dataloader:
  batch_size: 64
  shuffle: true
  num_workers: 0

##### MODEL CONFIGURATION #####
model_info:
  saving_params:
    model_save_name: DeepESD_BerGamma
  loss_params:
    name: NLLBerGammaLoss
    module: deep4production.deep.loss
    kwargs:
      threshold: 0.999
      ignore_nans: True
  model_params:
    name: DeepESD
    module: deep4production.deep.models.cnn.DeepESD
    kwargs:
      x_shape: [15, 16, 16]    # (C, H, W) — use d4p-inspect to confirm
      y_shape: [1, 128, 128]   # (C, H, W)
      f_shape: [1, 128, 128]
      filters: [50, 25, 10]
      kernel_size: 3
      loss_function_name: NLLBerGammaLoss
  training_params:
    num_epochs: 1000
    patience_early_stopping: 30
    optimizer_params:
      lr: 0.0001
```

> **Mixing formats.** If you have some Zarr stores in d4p format and others in anemoi format, set `format: d4p` or `format: anemoi` independently on each block. The default is `format: auto`, which auto-detects the format from Zarr metadata.

Launch training:

```bash
d4p-train ./training/configs/deepesd_anemoi.yaml
```

Training output, model saving, early stopping, and optional MLflow integration are identical to the companion notebook. Refer to Section 6 of that notebook for a full description.

______________________________________________________________________

## 7. Run Inference with `d4p-downscale`

The inference configuration also requires only `format: anemoi` in the `input_data` block:

```yaml
# File: ./inference/configs/deepesd_anemoi.yaml

id_dir: ./outputs/deepesd

input_data:
  paths:
    - ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr
  format: anemoi   # <-- tells d4p to open this Zarr via AnemoiZarrStore
  years: [1980]
  load_in_memory: true

graph: null
ensemble_size: 2

model_file: DeepESD_BerGamma_best.pt

saving_info:
  file: 1980.nc
  template: null
  formatting: null

inference_params:    # Forwarded as **kwargs to downscaler.downscale()
  batch_size: 4
  amp_dtype: null    # 'bfloat16' (Ampere+) / 'float16' / null
  compile: false
```

Run inference:

```bash
d4p-downscale ./inference/configs/deepesd_anemoi.yaml
```

The output NetCDF file `./outputs/deepesd/predictions/1980.nc` has the same structure as in the companion notebook and can be opened with `xarray` identically.

______________________________________________________________________

## 8. Visualization

The visualization step is identical to the companion DeepESD notebook. Use `plot_date_from_1D_spatial_field` from `deep4production.viz.spatial`:

```python
import xarray as xr
from deep4production.viz.spatial import plot_date_from_1D_spatial_field

kwargs = {
    "date": "1980-01-01",
    "vmin": 0,
    "vmax": 10,
    "set_extent": [5, 15, 44, 48],
    "central_longitude": 0,
    "cbar_label": "Precipitation (mm)",
    "titles": ["Groundtruth (RCM)", "Prediction (DeepESD)", "Difference"],
    "diff": True,
    "vminDiff": -5,
    "vmaxDiff": 5,
    "cmapDiff": "BrBG",
}

tgt = xr.open_dataset("./source_files/data_zenodo/ALPS_domain/train/ESD_pseudo_reality/target/pr_tasmax_CNRM-CM5_1961-1980.nc")
tgt = tgt.stack(point=("y", "x"))
tgt["time"] = tgt.time.dt.floor("D")

prd = xr.open_dataset("./outputs/deepesd/predictions/1980.nc")
prd = prd.isel(member=0)
prd["time"] = prd.time.dt.floor("D")

var = "pr"
kwargs.update({"data": [tgt[var], prd[var]]})

fig = plot_date_from_1D_spatial_field(**kwargs)
```

______________________________________________________________________

## 9. Summary: d4p-create vs. anemoi-datasets

| Step | d4p workflow | anemoi workflow |
|------|-------------|-----------------|
| Build predictor Zarr | `d4p-create UPSRCM.yaml` | `anemoi-datasets create UPSRCM_anemoi.yaml output.zarr` |
| Build predictand Zarr | `d4p-create RCM.yaml` (includes static `orog`) | `anemoi-datasets create RCM_anemoi.yaml output.zarr` (time-varying only) |
| Inspect | `d4p-inspect output.zarr` | `d4p-inspect output.zarr` (unchanged) |
| Training YAML | no `format` key (d4p default) | `format: anemoi` in each block |
| Inference YAML | no `format` key | `format: anemoi` in `input_data` |
| Train | `d4p-train deepesd.yaml` | `d4p-train deepesd_anemoi.yaml` |
| Downscale | `d4p-downscale deepesd.yaml` | `d4p-downscale deepesd_anemoi.yaml` |
| Model outputs | same | same |

You have now completed the full deep4production workflow using anemoi-datasets as the data-ingest backend with the DeepESD model for the CORDEX-BENCH Alps case study:

- Built AI-ready Zarr stores with `anemoi-datasets create`
- Inspected the anemoi Zarr stores with `d4p-inspect`
- Trained a DeepESD model with `d4p-train` reading anemoi Zarrs
- Performed inference with `d4p-downscale` reading anemoi Zarrs

______________________________________________________________________

## 10. References

- [CORDEX-BENCH GitHub](https://github.com/WCRP-CORDEX/ml-benchmark)
- [CORDEX-BENCH Zenodo](https://zenodo.org/records/17957264)
- [anemoi-datasets](https://github.com/ecmwf/anemoi-datasets)
- [anemoi-datasets documentation](https://anemoi-datasets.readthedocs.io)
- [DeepESD](https://gmd.copernicus.org/articles/15/6747/2022/)
- [deep4production](https://github.com/SantanderMetGroup/deep4production)
