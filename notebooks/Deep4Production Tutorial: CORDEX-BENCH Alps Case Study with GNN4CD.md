# Deep4Production Tutorial: CORDEX-BENCH Alps Case Study with GNN4CD

This tutorial demonstrates how to use the [deep4production](https://github.com/SantanderMetGroup/deep4production) framework for **graph-based** climate downscaling using **GNN4CD** (Blasone et al. 2025). Instead of treating the predictand grid as a 2-D image (as DeepESD, CPMGEM, and ResDiff do), GNN4CD builds a **bipartite graph** linking low-resolution and high-resolution grid cells and propagates information through learnable message passing.

This is particularly attractive when:

- The high-resolution grid is **irregular** (e.g. a sub-region of a global grid, or station data) — GNN4CD does not require a regular `(H, W)` raster.
- You need to evaluate the trained model on a **different geographical domain or grid** — the same network can be reused with a freshly built graph at inference time.

> If you are new to `deep4production`, please go through the [DeepESD tutorial](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md) first — sections 1 through 5 (project structure, data download, dataset creation, inspection) are summarized briefly here.

______________________________________________________________________

## 1. Introduction

The CLI tools (`d4p-create`, `d4p-inspect`, `d4p-train`, `d4p-downscale`) and project structure are described in the [DeepESD tutorial, section 1](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md#1-introduction).

GNN4CD swaps the convolutional backbone of DeepESD for a graph neural network and introduces three deep4production features absent from CNN-based recipes:

1. A **`graph` block** in the YAML — `deep4production` builds an `edge_index` from the predictor / predictand lat-lon coordinates the first time the trainer runs and caches it to disk for reuse.
1. **Lagged predictors** (`num_lagged: 1`) — a small recurrent encoder summarises the previous days' predictor states before message passing.
1. **`transform_to_2D: false`** for both predictors and predictands — the grid is treated as a flat list of nodes, not a 2-D image.

______________________________________________________________________

## 2. Case study: CORDEX-BENCH

Same simplified CORDEX-BENCH Alps configuration as the other tutorials, with two GNN-specific tweaks:

- **AI-model backbone:** **GNN4CD** (graph neural network with bipartite low-res ↔ high-res message passing)
- **Loss function:** `Asym` — MAE + a CDF-weighted term that penalises under-prediction of extreme precipitation
- **Predictors:** UPSRCM (16 × 16 = 256 nodes), 15 variables, with **1 lagged** day
- **Predictands:** RCM (128 × 128 = 16 384 nodes) precipitation `pr`, transformed by `log1p`
- **Training:** 1961-1978 · **Validation:** 1979, 1980 (note: this is a different split from DeepESD/CPMGEM/ResDiff — keep it in mind when comparing diagnostics across recipes)

______________________________________________________________________

## 3. Download CORDEX-BENCH Alps Data

Identical to the DeepESD tutorial — see [section 3 there](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md#3-download-cordex-bench-alps-data).

______________________________________________________________________

## 4. Prepare AI-Ready Datasets with `d4p-create`

Identical to the DeepESD tutorial. GNN4CD reads the **same Zarr files** (`UPSRCM_1961-1980.zarr` and `RCM_1961-1980.zarr`); the lat-lon coordinates stored in those Zarrs are what the graph builder uses to compute neighbour relationships. See [section 4 of the DeepESD tutorial](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md#4-prepare-ai-ready-datasets-with-d4p-create).

______________________________________________________________________

## 5. Inspect the Zarr Datasets with `d4p-inspect`

```bash
d4p-inspect ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr   # Predictors
d4p-inspect ./AI_ready_datasets/files/RCM_1961-1980.zarr      # Predictands
```

In addition to the per-variable statistics, GNN4CD also relies on the `lats` / `lons` arrays stored in the zarr metadata to build the graph. Check that they cover the expected domain.

______________________________________________________________________

## 6. Train GNN4CD with `d4p-train`

GNN4CD differs from DeepESD in five places:

1. **Custom trainer (`d4p_trainer`).** `trainer_gnn4cd` reuses the cached graph each epoch and feeds the network with lagged sequences.
1. **`graph` block.** Builds the bipartite graph from the lat-lon coordinates of the predictor and predictand zarr files. The first run computes and caches the `edge_index.pt`; subsequent runs reuse it.
1. **Predictand operator (`log1p`).** Precipitation is transformed by `y → log(1 + y)` before being fed to the loss; this stabilises training on the heavy-tailed distribution. The inverse `expm1` is applied automatically at inference.
1. **`transform_to_2D: false` and `num_lagged: 1`.** The data are served as flat node sequences, with each sample being the current day plus one lag.
1. **`batch_size: 1`.** The graph topology is static and contains every node, so a single batch per step is the natural choice (matching the original GNN4CD implementation).

`./gnn4cd_asym/train.yaml`:

```yaml
##### GENERAL INFO #####
run_ID: gnn4cd_asym
output_dir: .
overwrite: true

##### TRAINER + GRAPH CACHE #####
d4p_trainer:
  name: trainer_custom
  module: deep4production.core.trainers.trainer_gnn4cd
  kwargs:
    edge_index_path: ./gnn4cd_asym/outputs/aux_files/edge_index.pt   # built on first run

##### TRAINING DATA CONFIGURATION #####
data:
  load_in_memory: true
  training_period: [1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969,
                    1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978]
  validation_period: [1979, 1980]

  predictors:
    paths:
      - ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr
    variables: null            # null → use all variables present in the zarr
    normalizer:
      path_reference: ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr
      default: mean_std
    transform_to_2D: false     # GNN consumes flat node sequences
    num_lagged: 1              # current day + 1 lag → recurrent encoder

  predictands:
    paths:
      - ./AI_ready_datasets/files/RCM_1961-1980.zarr
    variables:
      - pr
    transform_to_2D: false
    operator:
      pr: log1p                # y → log(1 + y); inverse applied at inference

##### DATA LOADER CONFIGURATION #####
dataloader:
  batch_size: 1                # graph is static → one full graph per step
  shuffle: false
  num_workers: 0

##### GRAPH CONSTRUCTION #####
# Built on first run from the lat/lon coords of the two zarrs and cached at
# d4p_trainer.kwargs.edge_index_path. Pass `path: edge_index.pt` at inference
# to reuse the cached graph.
graph:
  name: build_graph
  module: deep4production.deep.models.gnn.GNN4CD
  kwargs:
    data_high: ./AI_ready_datasets/files/RCM_1961-1980.zarr
    data_low:  ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr
    # Paper default (Blasone et al. 2025): 8 distinct high-high neighbours per
    # node (self excluded), bidirectional edges. 4 here is a faster departure.
    nearest_neighbours_high_to_high: 8
    nearest_neighbours_low_to_high: 4

##### TRAINING CONFIGURATION #####
model_info:
  saving_params:
    model_save_name: gnn4cd_asym

  # Asymmetric loss: MAE + CDF-weighted penalty for under-prediction of extremes.
  loss_params:
    name: Asym
    module: deep4production.deep.loss
    kwargs:
      ref_path: ./AI_ready_datasets/files/RCM_1961-1980.zarr
      var: pr
      type: full               # one Gamma fit per gridpoint over the whole period
      asym_path: ./gnn4cd_asym/outputs/aux_files/
      asym_weight: 1
      cdf_pow: 2
      appendix: null
      ignore_nans: true

  model_params:
    name: GNN4CD
    module: deep4production.deep.models.gnn.GNN4CD
    kwargs:
      c_low: 15                # low-res predictor channels
      c_high: null             # no high-res forcings (e.g. orography) here
      c_rnn_out: 15
      num_layers_rnn: 2
      num_lagged_predictors: 1
      channels_downscaler_low_in: 8
      channels_downscaler_out: 16
      channels_downscaler_base: 8
      pred_dim: 1              # single high-res target variable

  training_params:
    num_epochs: 500
    patience_early_stopping: 20
    optimizer_params:
      lr: 0.0001
```

Train with:

```bash
d4p-train ./gnn4cd_asym/train.yaml
```

> 💡 **First-run cost.** On the very first run the trainer builds the graph (kNN search over the predictand and predictor lat-lon coordinates) and writes it to `./gnn4cd_asym/outputs/aux_files/edge_index.pt`. Subsequent runs (resuming training, hyperparameter sweeps) reuse this file — only the kNN search is cached, so changing `nearest_neighbours_*` *will* trigger a rebuild.

Below is an example of training output:

______________________________________________________________________

### Enabling MLflow

The MLflow block is identical to the [DeepESD tutorial section](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md#enabling-mlflow-in-deep4production); useful tags for filtering: `model: gnn4cd`, `loss: asym`, `architecture: graph`.

A handy diagnostic to add for graph models — total wall-clock per epoch can be high on large grids; `compute_diagnostics_every_n_epochs: 5–10` keeps overhead low.

______________________________________________________________________

## 7. Run Inference with `d4p-downscale`

The inference YAML reuses the cached graph by setting `graph.path` to the file written during training. The model is then evaluated on the requested years.

`./gnn4cd_asym/inference.yaml`:

```yaml
run_ID: gnn4cd_asym
output_dir: .

d4p_downscaler:
  name: d4p_downscaler_custom
  module: deep4production.core.downscalers.downscaler_gnn4cd

input_data:
  paths:
    - ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr
  years: [1980]
  load_in_memory: true

ensemble_size: 1
model_file: gnn4cd_asym_best.pt   # relative to id_dir/outputs/models/

graph:
  path: edge_index.pt              # reuse the cached graph (relative to id_dir/outputs/aux_files/)

  ## Want to apply the model to a different graph (e.g. a different domain)?
  ## Replace the block above with the snippet below — `build_graph` will run
  ## at inference time and the trained weights will be applied to the new graph.
  # path: null
  # name: build_graph
  # module: deep4production.deep.models.gnn.GNN4CD
  # kwargs:
  #   data_high: ./AI_ready_datasets/files/RCM_2080-2099.zarr
  #   data_low:  ./AI_ready_datasets/files/UPSRCM_2080-2099.zarr
  #   nearest_neighbours_high_to_high: 8
  #   nearest_neighbours_low_to_high: 4

saving_info:
  file: 1980.nc                    # saved at id_dir/outputs/predictions/
  template: null

inference_params:                  # Forwarded as **kwargs to downscaler.downscale()
  batch_size: 4
  amp_dtype: null                  # 'bfloat16' (Ampere+) / 'float16' / null
  compile: false
```

Run with:

```bash
d4p-downscale ./gnn4cd_asym/inference.yaml
```

Below is an example of inference output:

The downscaler automatically applies the inverse `log1p` operator (`expm1`) before saving, so predictions are written in physical units (mm/day):

> 🔁 **Cross-domain inference.** Because the network only depends on local graph connectivity (not on a fixed `(H, W)` raster), the same trained `gnn4cd_asym_best.pt` can be evaluated on a different domain by pointing `graph.data_high` / `graph.data_low` at a different pair of zarrs. This is the GNN counterpart of the "drop-in retraining" cost that CNN-based downscalers usually pay.

______________________________________________________________________

## 8. Visualization

```python
import xarray as xr
import numpy as np
from deep4production.viz.spatial import plot_date_from_1D_spatial_field

kwargs = {
    "date": "1980-01-01",
    "vmin": 0,
    "vmax": 10,
    "set_extent": [5, 15, 44, 48],
    "central_longitude": 0,
    "cbar_label": "Precipitation (mm)",
    "titles": ["Groundtruth (RCM)", "Prediction (GNN4CD)", "Difference"],
    "diff": True,
    "vminDiff": -5,
    "vmaxDiff": 5,
    "cmapDiff": "BrBG",
}

tgt = xr.open_dataset(
    "./source_files/data_zenodo/ALPS_domain/train/ESD_pseudo_reality/target/"
    "pr_tasmax_CNRM-CM5_1961-1980.nc"
)
tgt = tgt.stack(point=("y", "x"))
tgt['time'] = tgt.time.dt.floor('D')

prd = xr.open_dataset("./gnn4cd_asym/outputs/predictions/1980.nc")
prd = prd.isel(member=0)
prd['time'] = prd.time.dt.floor('D')

var = "pr"
kwargs.update({"data": [tgt[var], prd[var]]})

fig = plot_date_from_1D_spatial_field(**kwargs)
```

______________________________________________________________________

## 9. Summary

You have now run a full graph-based downscaling workflow on CORDEX-BENCH Alps using GNN4CD:

- Reused the AI-ready datasets created in the DeepESD tutorial.
- Built a bipartite low-res ↔ high-res graph from the zarr lat-lon metadata (cached automatically).
- Trained a GNN with the Asym loss to handle precipitation extremes.
- Performed inference reusing the cached graph, with the option to swap the graph at inference for cross-domain evaluation.

For an image-based deterministic baseline, see the [DeepESD tutorial](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md). For probabilistic alternatives based on diffusion, see the [CPMGEM](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20CPMGEM/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20CPMGEM.md) and [ResDiff](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20ResDiff/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20ResDiff.md) tutorials.

______________________________________________________________________

## 10. References

- [CORDEX-BENCH GitHub](https://github.com/WCRP-CORDEX/ml-benchmark)
- [CORDEX-BENCH Zenodo](https://zenodo.org/records/17957264)
- Blasone et al. 2025 — *GNN4CD: graph neural networks for climate downscaling*
