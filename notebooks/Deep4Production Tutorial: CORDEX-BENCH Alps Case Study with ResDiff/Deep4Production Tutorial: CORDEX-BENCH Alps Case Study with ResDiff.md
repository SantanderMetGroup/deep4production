# Deep4Production Tutorial: CORDEX-BENCH Alps Case Study with ResDiff

This tutorial demonstrates how to use the [deep4production](https://github.com/SantanderMetGroup/deep4production) framework for **residual diffusion** climate downscaling using **ResDiff** (CorrDiff-style, Mardani et al. 2023). ResDiff is a **two-stage** model:

1. A **deterministic regressor** (a SongUNet trained as a regression model) predicts the conditional mean `ŷ = f(x)`.
2. An **EDM-preconditioned diffusion model** generates the **residual** `r = y − ŷ` conditioned on the low-res predictors *and* on the regression mean `ŷ`. The final downscaled prediction is `ŷ + r̂`.

Splitting the problem this way lets the regressor capture the smooth, large-scale signal cheaply while the diffusion model concentrates capacity on small-scale stochastic structure — this is the recipe used by NVIDIA's CorrDiff.

> If you are new to `deep4production`, please go through the [DeepESD tutorial](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md) first — sections 1 through 5 (project structure, data download, dataset creation, inspection) are summarized briefly here.

---

## 1. Introduction

**deep4production** ships a residual-diffusion trainer (`trainer_resdiff`) and a custom pydataset (`pydataset_resdiff`) that:

1. Loads the pre-trained regressor checkpoint.
2. On first run, computes residuals on every training/validation date and caches them to a Zarr file (subsequent runs skip this step).
3. Serves `(r, c_low, c_high)` triples to the EDM-preconditioned SongUNet.

The CLI tools (`d4p-create`, `d4p-inspect`, `d4p-train`, `d4p-downscale`) and project structure are described in the [DeepESD tutorial, section 1](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md#1-introduction).

---

## 2. Case study: CORDEX-BENCH

Same simplified CORDEX-BENCH Alps configuration as the DeepESD/CPMGEM tutorials:

* **Domain:** Central Europe (Alps)
* **AI-model backbone:** **ResDiff** (CorrDiff-style — deterministic SongUNet regressor + EDM-preconditioned residual diffusion)
* **Loss function (residual stage):** EDM-weighted denoising score matching
* **Loss function (regressor stage):** MSE
* **Predictors:** UPSRCM (16 × 16, 15 variables: `u/v/t/q/z` at 850/700/500 hPa)
* **Predictands:** RCM (128 × 128) precipitation `pr`
* **Training:** 1961-1979 (excl. 1967, 1975) · **Validation:** 1967, 1975 · **Test:** 1980

---

## 3. Download CORDEX-BENCH Alps Data

Identical to the DeepESD tutorial — see [section 3 there](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md#3-download-cordex-bench-alps-data).

---

## 4. Prepare AI-Ready Datasets with `d4p-create`

Identical to the DeepESD tutorial — both stages of ResDiff read from the **same Zarr files** (`UPSRCM_1961-1980.zarr` and `RCM_1961-1980.zarr`). See [section 4 of the DeepESD tutorial](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md#4-prepare-ai-ready-datasets-with-d4p-create).

---

## 5. Inspect the Zarr Datasets with `d4p-inspect`

```bash
d4p-inspect ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr   # Predictors
d4p-inspect ./AI_ready_datasets/files/RCM_1961-1980.zarr      # Predictands
```

Use the reported predictor channel count to fill `cond_low_channels` in both YAMLs below.

---

## 6. Train ResDiff — a two-step workflow

### 6.1. Step 1 — Train the deterministic SongUNet regressor

The regressor is a SongUNet trained *deterministically*: the noisy-input slot is filled with zeros, the noise label `t` is fixed at zero, and the network conditions only on the low-res predictor (via `cond_low`). This reuses the same backbone class that the diffusion stage will use, which keeps train/inference machinery in lockstep.

`./training/configs/song_unet_det.yaml`:

```yaml
##### GENERAL INFO #####
run_ID: song_unet_det
output_dir: ./outputs
overwrite: true


##### TRAINER SELECTION #####
# Deterministic SongUNet trainer: bypasses diffusion by setting x_in=0, t=0
# and conditioning the UNet exclusively through cond_low (the low-res predictor).
d4p_trainer:
  name: trainer_custom
  module: deep4production.core.trainers.trainer_song_unet_det


##### TRAINING DATA CONFIGURATION #####
data:
  load_in_memory: true
  training_period: [1961, 1962, 1963, 1964, 1965, 1966, 1968, 1969, 1970, 1971,
                    1972, 1973, 1974, 1976, 1977, 1978, 1979, 1980]
  validation_period: [1967, 1975]

  predictors:
    paths:
      - ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr
    variables: [u_850, u_700, u_500, v_850, v_700, v_500,
                t_850, t_700, t_500, q_850, q_700, q_500,
                z_850, z_700, z_500]
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
    variables:
      - pr
    normalizer:
      path_reference: ./AI_ready_datasets/files/RCM_1961-1980.zarr
      default: std
    transform_to_2D: true


##### DATA LOADER CONFIGURATION #####
dataloader:
  batch_size: 16
  shuffle: true
  num_workers: 1


##### MODEL CONFIGURATION #####
model_info:
  saving_params:
    model_save_name: SongUNet_det
    save_every_n_epochs: 50

  loss_params:
    name: MseLoss
    module: deep4production.deep.loss
    kwargs:
      ignore_nans: false

  model_params:
    name: SongUNet
    module: deep4production.deep.models.unet.song_unet
    kwargs:
      in_channels: 1             # C_y — number of predictand variables
      cond_low_channels: 15      # C_x — predictor channels (d4p-inspect)
      cond_high_channels: 0      # no high-res conditioning in the deterministic stage
      nf: 128
      ch_mult: [1, 2, 2, 2]
      num_res_blocks: 4
      attn_at_levels: [3]
      dropout: 0.1
      fir: true
      fir_kernel: [1, 3, 3, 1]
      skip_rescale: true
      progressive_input: true
      cond_upsample: nearest
      spatial_pe_freqs: 1        # NeRF-style 4-channel spatial PE

  training_params:
    amp: false
    compile: false
    num_epochs: 1000
    patience_early_stopping: 30
    optimizer_params:
      lr: 0.0001
```

Train it:

```bash
d4p-train ./training/configs/song_unet_det.yaml
```

The best checkpoint is written to `./outputs/song_unet_det/models/SongUNet_det_best.pt`. **You will need this path for step 2.**

Below is an example of training output:

![d4p-train-regressor](./images/d4p-train-output-regressor.png)

### 6.2. Step 2 — Train the residual diffusion model

The residual stage adds three things on top of the regressor recipe:

1. **`d4p_pydataset`** — `pydataset_resdiff` runs the regressor on every (training and validation) date, computes `r = y − ŷ`, and caches the residual + the regression mean to a Zarr at `residuals.path` (suffixed with `_training.zarr` / `_validation.zarr`). On subsequent runs (e.g. resuming training) the cache is reused, so this one-time cost is only paid once.

2. **`d4p_trainer`** — `trainer_resdiff` implements the EDM training loop: log-normal noise sampling, EDM preconditioning (handled inside `EDMPrecond`), and EDM-weighted DSM loss.

3. **High-res conditioning (`cond_high_channels: 1`)** — the regressor's output `ŷ` is passed back to the diffusion U-Net as a *high-res* conditioning channel, in addition to the standard low-res predictor stream.

`./training/configs/resdiff.yaml`:

```yaml
##### GENERAL INFO #####
run_ID: resdiff
output_dir: ./outputs
overwrite: true


##### TRAINER SELECTION #####
# EDM residual-diffusion trainer (CorrDiff-style, Mardani et al. 2023).
# Predicts the residual r = y - mean_pred where mean_pred is the output of a
# separately trained deterministic regressor.
d4p_trainer:
  name: trainer_custom
  module: deep4production.core.trainers.trainer_resdiff


##### CUSTOM PYDATASET #####
# Residual pydataset: loads predictors + predictands, runs the deterministic
# regressor at init, caches residuals + regression means to a zarr, and
# serves (r, c_low, c_high) triples. c_high is the regression mean.
d4p_pydataset:
  name: pydataset_custom
  module: deep4production.core.pydatasets.pydataset_resdiff
  kwargs:
    # Path to the deterministic regressor checkpoint (trained in step 6.1).
    path_regressor: ./outputs/song_unet_det/models/SongUNet_det_best.pt
    add_pred_mean: true        # feed ŷ as cond_high
    add_context_lowres: true   # feed raw predictors as cond_low
    residuals:
      path: ./outputs/resdiff/aux_files/residuals.zarr
      template: ./templates/pr_template.nc


##### TRAINING DATA CONFIGURATION #####
data:
  load_in_memory: true
  training_period: [1961, 1962, 1963, 1964, 1965, 1966, 1968, 1969, 1970, 1971,
                    1972, 1973, 1974, 1976, 1977, 1978, 1979, 1980]
  validation_period: [1967, 1975]

  predictors:
    paths:
      - ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr
    variables: [u_850, u_700, u_500, v_850, v_700, v_500,
                t_850, t_700, t_500, q_850, q_700, q_500,
                z_850, z_700, z_500]
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
    variables:
      - pr
    normalizer:
      path_reference: ./AI_ready_datasets/files/RCM_1961-1980.zarr
      default: std
    transform_to_2D: True


##### DATA LOADER CONFIGURATION #####
dataloader:
  batch_size: 16
  shuffle: true
  num_workers: 4


##### MODEL CONFIGURATION #####
model_info:
  saving_params:
    model_save_name: ResDiff

  # EDM-weighted DSM on the *denoised* prediction D_theta returned by the
  # preconditioner. λ(σ) = (σ_d² + σ²) / (σ_d · σ)²; σ_data must match the
  # preconditioner's value below.
  loss_params:
    name: WeightedDenoisingScoreMatchingLoss
    module: deep4production.deep.loss
    kwargs:
      ignore_nans: false
      sigma_data: 0.5

  # EDM preconditioner wrapping the SongUNet backbone.
  model_params:
    name: build_edm_model
    module: deep4production.deep.models.diffusion.edm_precond
    kwargs:
      sigma_data: 0.5             # must match loss.sigma_data
      backbone:
        module: deep4production.deep.models.unet.song_unet
        name: SongUNet
        kwargs:
          in_channels: 1            # C_r — channels of the residual
          cond_low_channels: 15     # C_x — predictor channels
          cond_high_channels: 1     # C_yhat — regression mean as high-res context
          nf: 128
          ch_mult: [1, 2, 2, 2]
          num_res_blocks: 4
          attn_at_levels: [3]
          dropout: 0.13             # CorrDiff paper value
          fir: true
          fir_kernel: [1, 3, 3, 1]
          skip_rescale: true
          progressive_input: true
          cond_upsample: fir
          spatial_pe_freqs: 1       # 4-channel NeRF-style spatial PE

  training_params:
    num_epochs: 20
    patience_early_stopping: 5
    optimizer_params:
      lr: 0.0002
    ema_decay: 0.9999
    grad_clip: 1.0

    scheduler_params:
      type: LambdaLR
      lr_lambda: rampup_expdecay
      kwargs:
        base_lr: 0.0002
        rampup_steps: 5000
        decay_rate: 1.0
        decay_interval: 1
        terminal_value: 1.0e-6

    # EDM log-normal noise sampling (Karras et al. 2022).
    kwargs:
      noise_params:
        P_mean: -1.2
        P_std:  1.2
        sigma_min: 0.002
        sigma_max: 80.0
```

Train it:

```bash
d4p-train ./training/configs/resdiff.yaml
```

> 💡 **First-run cost.** On the very first run the trainer will iterate over every training/validation date, run the regressor, and write `residuals_training.zarr` + `residuals_validation.zarr`. This can take several minutes. Subsequent runs (e.g. when changing a hyperparameter or resuming training) reuse those Zarr files, so the cost is paid only once per dataset split.

Below is an example of training output:

![d4p-train-resdiff](./images/d4p-train-output-resdiff.png)

---

### Enabling MLflow

The MLflow block is identical to the [DeepESD tutorial section](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md#enabling-mlflow-in-deep4production); useful tags for filtering: `model: resdiff`, `loss: edm_dsm`, `regressor: song_unet_det`.

As with CPMGEM, set `compute_diagnostics_every_n_epochs` higher (e.g. `5`) — diagnostic samples require an EDM Heun sampler rollout per sample, which is much costlier than a deterministic forward pass.

---

## 7. Run Inference with `d4p-downscale`

The ResDiff downscaler:

1. Loads the regressor from `path_regressor` (stored in the training metadata, optionally overridden in the inference YAML).
2. For each input date: runs `regressor(x) → ŷ`, then samples a residual `r̂` from the EDM Heun sampler conditioned on `(x, ŷ)`.
3. Returns `ŷ + r̂` as the final downscaled field (still in normalised space; the base downscaler post-processing denormalises).

`./inference/configs/resdiff.yaml`:

```yaml
##### GENERAL INFO #####
id_dir: ./outputs/resdiff


##### INPUT DATA #####
input_data:
  paths:
    - ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr
  years: [1980]
  load_in_memory: true


##### ENSEMBLE SIZE #####
# CorrDiff is stochastic — each reverse-diffusion trajectory draws fresh
# Gaussian noise, so ensemble_size > 1 produces independent realisations.
ensemble_size: 5

graph: null


##### MODEL CHECKPOINT #####
# Metadata inside the checkpoint holds:
#   model_params.name   = build_edm_model
#   model_params.module = deep4production.deep.models.diffusion.edm_precond
#   model_params.kwargs = { sigma_data, backbone: { module, name, kwargs } }
# load_model() re-invokes the factory, so the preconditioner + backbone are
# reconstructed identically at inference without any extra wiring here.
model_file: ResDiff_best.pt


##### DOWNSCALER CLASS #####
d4p_downscaler:
  name: downscaler_custom
  module: deep4production.core.downscalers.downscaler_resdiff
  kwargs:
    # Optional: override the regressor path stored in training metadata
    # (useful if the checkpoint was moved after training).
    # path_regressor: /new/path/to/SongUNet_det_best.pt
    sampling_params:
      # EDM Heun sampler (Karras et al. 2022).
      num_steps: 18              # paper default for CorrDiff
      # sigma_min / sigma_max are read automatically from training metadata.
      rho: 7.0                   # EDM noise-schedule curvature
      # Stochastic churn parameters (Alg. 2 in Karras et al.).
      # S_churn=0 → deterministic Heun sampler (recommended baseline).
      S_churn: 0.0
      S_min: 0.0
      S_max: .inf
      S_noise: 1.0


##### OUTPUT #####
saving_info:
  file: 1980.nc
  template: null
  formatting: null
```

Run with:

```bash
d4p-downscale ./inference/configs/resdiff.yaml
```

Below is an example of inference output:

![d4p-downscale-1](./images/d4p-downscale-output.png)

> ⚡ **Why ResDiff is faster than CPMGEM at inference.** CPMGEM uses ~1000 reverse-SDE steps; the EDM Heun sampler typically needs only ~18, because the regressor already provides the smooth large-scale signal and only the small-scale residual structure needs to be sampled. The total cost per ensemble member is roughly `1 × cost(regressor) + 18 × cost(diffusion UNet)`.

The output NetCDF has shape `(member, time, point)` — same layout as CPMGEM:

![d4p-downscale-2](./images/d4p-downscale-pred.png)

---

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
    "titles": ["Groundtruth (RCM)", "Prediction (ResDiff, member 0)", "Difference"],
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

prd = xr.open_dataset("./outputs/resdiff/predictions/1980.nc")
prd = prd.isel(member=0)
prd['time'] = prd.time.dt.floor('D')

var = "pr"
kwargs.update({"data": [tgt[var], prd[var]]})

fig = plot_date_from_1D_spatial_field(**kwargs)
```

![figure](./images/resdiff_1980-01-01.png)

### Decomposing the prediction

Because ResDiff is residual-based, you can inspect the regression mean and the diffusion residual *separately* to see which part of the field comes from each stage. Run inference once with the regressor only (using its own inference YAML) and once with the full ResDiff downscaler, then plot:

* `ŷ` — smooth, deterministic — comes from the regressor.
* `r̂ = ResDiff − ŷ` — small-scale stochastic detail — comes from the diffusion stage.

This is one of the main reasons to prefer the residual formulation: the two contributions are interpretable and you can blame failures on the appropriate stage.

---

## 9. Summary

You have now run the complete two-stage residual-diffusion workflow on CORDEX-BENCH Alps:

* Trained a deterministic SongUNet regressor (step 1).
* Cached the regressor's residuals to a Zarr (handled automatically by `pydataset_resdiff` on first run).
* Trained an EDM-preconditioned SongUNet to denoise the residual conditioned on `(x, ŷ)` (step 2).
* Generated **ensemble** high-resolution precipitation fields by running the regressor + EDM Heun sampler.

For a **direct** (single-stage) diffusion workflow without the regressor, see the [CPMGEM tutorial](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20CPMGEM/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20CPMGEM.md).

---

## 10. References

* [CORDEX-BENCH GitHub](https://github.com/WCRP-CORDEX/ml-benchmark)
* [CORDEX-BENCH Zenodo](https://zenodo.org/records/17957264)
* Mardani et al. 2023 — *Generative Residual Diffusion Modeling for Km-Scale Atmospheric Downscaling* (CorrDiff)
* Karras et al. 2022 — *Elucidating the Design Space of Diffusion-Based Generative Models* (EDM)
* Song et al. 2021 — *Score-based generative modeling through SDEs* (NCSN++ / SongUNet backbone)
