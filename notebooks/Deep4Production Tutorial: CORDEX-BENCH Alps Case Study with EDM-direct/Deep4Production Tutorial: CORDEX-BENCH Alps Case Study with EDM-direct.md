# Deep4Production Tutorial: CORDEX-BENCH Alps Case Study with EDM-direct

This tutorial demonstrates **EDM-direct** downscaling on the CORDEX-BENCH Alps domain. EDM-direct is a **hybrid** of the two diffusion models you have already met:

- it generates the **full high-resolution field directly** in a single stage — no regressor, no residual — exactly like [CPMGEM](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20CPMGEM/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20CPMGEM.md), but
- it uses the **EDM/CorrDiff diffusion formulation** (EDM preconditioning + λ(σ)-weighted denoising score matching + Heun sampler) of [ResDiff](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20ResDiff/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20ResDiff.md) instead of CPMGEM's sub-VP SDE.

In short: **EDM-direct is the EDM analogue of CPMGEM.**

> If you are new to `deep4production`, go through the [DeepESD tutorial](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md) (sections 1–5), then skim [CPMGEM](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20CPMGEM/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20CPMGEM.md) (direct workflow) and [ResDiff](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20ResDiff/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20ResDiff.md) (EDM machinery).

______________________________________________________________________

## 1. Where this model sits — the 2×2 study

EDM-direct is the other off-diagonal cell of the **2×2 decoupling study** that separates the **diffusion formulation** (sub-VP vs. EDM) from the **parameterization target** (direct full-field vs. residual-to-regressor):

|                       | Direct (predict full `y`)            | Residual to regressor                |
| --------------------- | ------------------------------------ | ------------------------------------ |
| **Sub-VP (CPMGEM)**   | CPMGEM                                | [CPMGEM-residual](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20CPMGEM-residual/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20CPMGEM-residual.md) |
| **EDM (CorrDiff)**    | **EDM-direct (this tutorial)**       | ResDiff                              |

EDM-direct shares the **target** with CPMGEM (full field) and the **formulation** with ResDiff (EDM), so it is the pivot for isolating which axis matters.

> **Normalization convention (important).** The clean-2×2 cells — **CPMGEM, CPMGEM-residual, EDM-direct** — all use the same precipitation preprocessing, `sqrt` then `minmax_neg1_1` (with `sigma_data ≈ 0.5` for the EDM cells), so the predictand transform is not a confound. **ResDiff is the exception**: it keeps CorrDiff's native `std` normalization, by design, as a faithful-CorrDiff baseline. So when EDM-direct is compared to ResDiff along the *target* axis, remember that comparison also carries a normalization difference (see section 9).

______________________________________________________________________

## 2. Case study: CORDEX-BENCH

- **Domain:** Central Europe (Alps)
- **AI-model backbone:** **EDM-direct** (EDM-preconditioned SongUNet, single-stage full-field generation)
- **Loss function:** EDM-weighted denoising score matching (λ(σ)-weighted)
- **Predictors:** UPSRCM (16 × 16, 15 variables: `u/v/t/q/z` at 850/700/500 hPa)
- **Predictands:** RCM (128 × 128) precipitation `pr`
- **Training:** 1961-1979 (excl. 1967, 1975) · **Validation:** 1967, 1975 · **Test:** 1980

______________________________________________________________________

## 3–5. Data download, dataset creation, inspection

Identical to the DeepESD/CPMGEM tutorials — EDM-direct reads from the same Zarr files (`UPSRCM_1961-1980.zarr`, `RCM_1961-1980.zarr`) and uses the **standard pydataset** (no residual cache, no regressor). See [DeepESD sections 3–5](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md#3-download-cordex-bench-alps-data).

```bash
d4p-inspect ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr   # Predictors
d4p-inspect ./AI_ready_datasets/files/RCM_1961-1980.zarr      # Predictands
```

______________________________________________________________________

## 6. Train EDM-direct with `d4p-train` (single stage)

Unlike ResDiff, EDM-direct is a **one-step** workflow — there is no regressor to train first. Relative to CPMGEM, only the diffusion engine changes:

1. **`d4p_trainer`** — `trainer_edm_direct`. It uses the **standard pydataset** `(x, y, f)` (so the base trainer's data wiring is unchanged), normalizes the full target field `y`, and runs the **EDM forward process** on it (`y_t = y + σ·z`, log-normal σ), predicting the denoised `D_θ`.

1. **EDM model + loss** — `build_edm_model` wraps the SongUNet in an `EDMPrecond`, and the loss is `WeightedDenoisingScoreMatchingLoss` — exactly as in ResDiff, but here the clean signal is the full field, not a residual.

1. **No standardization needed.** This is the key contrast with CPMGEM-residual: EDM absorbs the data scale through its `sigma_data` preconditioning, so the full normalized field (`sqrt(pr)` → `[−1,1]`, std ≈ `sigma_data` = 0.5) needs no extra rescaling. (This is precisely the scale-handling that sub-VP lacks.)

`./training/configs/edm_direct.yaml`:

```yaml
##### GENERAL INFO #####
run_ID: edm_direct
output_dir: ./outputs
overwrite: true


##### TRAINER SELECTION #####
# EDM diffusion generating the full field directly (EDM × direct cell).
# Standard pydataset (x, y, f) — no d4p_pydataset block, no regressor.
d4p_trainer:
  name: trainer_custom
  module: deep4production.core.trainers.trainer_edm_direct


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
    # sqrt(pr) → [-1, 1] keeps the field at ~unit scale, matched to sigma_data=0.5.
    operator:
      default: sqrt
    normalizer:
      path_reference: ./AI_ready_datasets/files/RCM_1961-1980.zarr
      default: minmax_neg1_1
    transform_to_2D: True


##### DATA LOADER CONFIGURATION #####
dataloader:
  batch_size: 16
  shuffle: true
  num_workers: 4


##### MODEL CONFIGURATION #####
model_info:
  saving_params:
    model_save_name: EDM_direct

  # EDM-weighted DSM on the denoised D_theta. sigma_data must match below.
  loss_params:
    name: WeightedDenoisingScoreMatchingLoss
    module: deep4production.deep.loss
    kwargs:
      ignore_nans: false
      sigma_data: 0.5

  # EDM preconditioner wrapping the SongUNet backbone, operating on the full
  # field. cond_high_channels: 0 (no regressor mean; set to C_f only if you add
  # forcings via a data.forcings block).
  model_params:
    name: build_edm_model
    module: deep4production.deep.models.diffusion.edm_precond
    kwargs:
      sigma_data: 0.5             # must match loss.sigma_data
      backbone:
        module: deep4production.deep.models.unet.song_unet
        name: SongUNet
        kwargs:
          in_channels: 1            # C_y — full target field channels
          cond_low_channels: 15     # C_x — predictor channels
          cond_high_channels: 0
          nf: 128
          ch_mult: [1, 2, 2, 2]
          num_res_blocks: 4
          attn_at_levels: [3]
          dropout: 0.13
          fir: true
          fir_kernel: [1, 3, 3, 1]
          skip_rescale: true
          progressive_input: true
          cond_upsample: fir
          spatial_pe_freqs: 1

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
d4p-train ./training/configs/edm_direct.yaml
```

> 🔬 **Fair-comparison note.** For a clean 2×2, keep the backbone capacity (`nf` / `ch_mult` / `num_res_blocks` / attention) matched to the cell you compare against: **CPMGEM** for the formulation axis (same direct target, same `sqrt + minmax` convention → clean). For the *target* axis the natural counterpart is the EDM-residual cell; **ResDiff** fills that slot but on `std` normalization, so a fully clean target-axis comparison would use a `sqrt + minmax` EDM-residual variant. The recipe above mirrors CPMGEM's backbone for direct comparability; adjust per your iso-params / iso-FLOPs budget.

______________________________________________________________________

## 7. Run Inference with `d4p-downscale`

The EDM-direct downscaler is single-stage: it draws the **full field** directly with the **EDM Heun sampler** (the same one ResDiff uses for its residual), conditioned only on the low-res predictors — no regressor.

`./inference/configs/edm_direct.yaml`:

```yaml
##### GENERAL INFO #####
id_dir: ./outputs/edm_direct


##### INPUT DATA #####
input_data:
  paths:
    - ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr
  years: [1980]
  load_in_memory: true


##### ENSEMBLE SIZE #####
ensemble_size: 5

graph: null


##### MODEL CHECKPOINT #####
# EDM-preconditioned SongUNet (build_edm_model); reconstructed from metadata.
model_file: EDM_direct_best.pt


##### DOWNSCALER CLASS #####
d4p_downscaler:
  name: downscaler_custom
  module: deep4production.core.downscalers.downscaler_edm_direct
  kwargs:
    sampling_params:
      num_steps: 18              # EDM/CorrDiff default
      # sigma_min / sigma_max read automatically from training metadata.
      rho: 7.0
      S_churn: 0.0               # 0 → deterministic Heun (recommended baseline)
      S_min: 0.0
      S_max: .inf
      S_noise: 1.0


##### OUTPUT #####
saving_info:
  file: 1980.nc
  template: null
  formatting: null


##### INFERENCE RUNTIME #####
inference_params:
  batch_size: 4
  amp_dtype: null
  compile: false
```

Run with:

```bash
d4p-downscale ./inference/configs/edm_direct.yaml
```

> ⚡ **Why EDM-direct is fast.** The EDM Heun sampler needs only ~18 steps versus CPMGEM's ~1000 reverse-SDE steps for the *same direct target*. This makes EDM-direct dramatically cheaper than CPMGEM at inference — a direct demonstration of the sampler/formulation axis the 2×2 study isolates. For quick smoke tests, set `ensemble_size: 1`.

The output NetCDF has shape `(member, time, point)`.

______________________________________________________________________

## 8. Visualization

Identical to the other diffusion tutorials — point `plot_date_from_1D_spatial_field` at `./outputs/edm_direct/predictions/1980.nc`:

```python
import xarray as xr
from deep4production.viz.spatial import plot_date_from_1D_spatial_field

kwargs = {
    "date": "1980-01-01",
    "vmin": 0, "vmax": 10,
    "set_extent": [5, 15, 44, 48],
    "central_longitude": 0,
    "cbar_label": "Precipitation (mm)",
    "titles": ["Groundtruth (RCM)", "Prediction (EDM-direct, member 0)", "Difference"],
    "diff": True, "vminDiff": -5, "vmaxDiff": 5, "cmapDiff": "BrBG",
}

tgt = xr.open_dataset(
    "./source_files/data_zenodo/ALPS_domain/train/ESD_pseudo_reality/target/"
    "pr_tasmax_CNRM-CM5_1961-1980.nc"
).stack(point=("y", "x"))
tgt['time'] = tgt.time.dt.floor('D')

prd = xr.open_dataset("./outputs/edm_direct/predictions/1980.nc").isel(member=0)
prd['time'] = prd.time.dt.floor('D')

kwargs.update({"data": [tgt["pr"], prd["pr"]]})
fig = plot_date_from_1D_spatial_field(**kwargs)
```

As with CPMGEM, aggregate across `member` for the ensemble mean and spread.

______________________________________________________________________

## 9. Summary

You have run the **EDM × direct** cell of the 2×2 study:

- Trained a single-stage EDM-preconditioned SongUNet to generate the **full field** directly — no regressor, no residual, no standardization (EDM `sigma_data` handles the data scale).
- Generated ensemble fields with the fast (~18-step) EDM Heun sampler.

Compare against [CPMGEM](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20CPMGEM/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20CPMGEM.md) (same direct target, sub-VP formulation — and the same `sqrt + minmax` convention, so the **formulation axis is cleanly isolated**). The EDM-residual corner is [ResDiff](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20ResDiff/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20ResDiff.md), but on `std` normalization, so an EDM-direct ↔ ResDiff (target-axis) comparison also carries the normalization difference; train a `sqrt + minmax` EDM-residual variant for a clean isolation. The three `sqrt + minmax` cells (CPMGEM, CPMGEM-residual, EDM-direct) are mutually confound-free.

______________________________________________________________________

## 10. References

- [CORDEX-BENCH GitHub](https://github.com/WCRP-CORDEX/ml-benchmark) · [Zenodo](https://zenodo.org/records/17957264)
- Karras et al. 2022 — *Elucidating the Design Space of Diffusion-Based Generative Models* (EDM)
- Mardani et al. 2023 — *Generative Residual Diffusion Modeling for Km-Scale Atmospheric Downscaling* (CorrDiff)
- Addison et al. 2024 — *Machine learning emulation of precipitation … using a diffusion model* (CPMGEM)
