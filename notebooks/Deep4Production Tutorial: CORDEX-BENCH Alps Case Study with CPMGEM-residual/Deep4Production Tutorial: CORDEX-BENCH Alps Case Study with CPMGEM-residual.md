# Deep4Production Tutorial: CORDEX-BENCH Alps Case Study with CPMGEM-residual

This tutorial demonstrates **CPMGEM-residual** downscaling on the CORDEX-BENCH Alps domain. CPMGEM-residual is a **hybrid** of the two diffusion models you have already met:

- it adopts the **two-stage residual** formulation of [ResDiff](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20ResDiff/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20ResDiff.md) (a deterministic regressor predicts the mean `ŷ`, and a diffusion model generates the residual `r = y − ŷ`), but
- it drives that residual with the continuous-time **sub-VP SDE** of [CPMGEM](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20CPMGEM/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20CPMGEM.md) instead of ResDiff's EDM formulation.

> If you are new to `deep4production`, go through the [DeepESD tutorial](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md) first (sections 1–5), then the [ResDiff tutorial](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20ResDiff/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20ResDiff.md) (the two-stage workflow is identical here — only the diffusion stage differs).

______________________________________________________________________

## 1. Where this model sits — the 2×2 study

CPMGEM-residual is one of the two off-diagonal cells of a **2×2 decoupling study** that separates the **diffusion formulation** (sub-VP vs. EDM) from the **parameterization target** (direct full-field vs. residual-to-regressor):

|                       | Direct (predict full `y`)                | Residual to regressor                       |
| --------------------- | ---------------------------------------- | ------------------------------------------- |
| **Sub-VP (CPMGEM)**   | CPMGEM                                    | **CPMGEM-residual (this tutorial)**         |
| **EDM (CorrDiff)**    | [EDM-direct](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20EDM-direct/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20EDM-direct.md) | ResDiff                                     |

The point of building both off-diagonal cells is that the formulation effect and the target effect can only be disentangled when all four cells exist.

> **Normalization convention (important).** To avoid the predictand transform becoming a confound, the clean-2×2 cells — **CPMGEM, CPMGEM-residual, EDM-direct** — all use the same precipitation preprocessing: `sqrt` then `minmax_neg1_1` (and, for the EDM cell, `sigma_data ≈ 0.5`). **ResDiff is the exception**: it deliberately keeps CorrDiff's native `std` normalization, so it serves as a *faithful-CorrDiff baseline* rather than the normalization-matched EDM-residual corner. A practical consequence for this tutorial: CPMGEM-residual needs **its own regressor trained with `sqrt + minmax_neg1_1`** — it does **not** share the `std`-trained regressor from the ResDiff tutorial.

______________________________________________________________________

## 2. Case study: CORDEX-BENCH

Same simplified CORDEX-BENCH Alps configuration as the other tutorials:

- **Domain:** Central Europe (Alps)
- **AI-model backbone:** **CPMGEM-residual** (deterministic SongUNet regressor + sub-VP SDE residual diffusion)
- **Loss (residual stage):** MSE on predicted noise ε
- **Loss (regressor stage):** MSE
- **Predictors:** UPSRCM (16 × 16, 15 variables: `u/v/t/q/z` at 850/700/500 hPa)
- **Predictands:** RCM (128 × 128) precipitation `pr`
- **Training:** 1961-1979 (excl. 1967, 1975) · **Validation:** 1967, 1975 · **Test:** 1980

______________________________________________________________________

## 3–5. Data download, dataset creation, inspection

Identical to the DeepESD/ResDiff tutorials — both stages read from the same Zarr files (`UPSRCM_1961-1980.zarr`, `RCM_1961-1980.zarr`). See [DeepESD sections 3–5](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md#3-download-cordex-bench-alps-data).

```bash
d4p-inspect ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr   # Predictors
d4p-inspect ./AI_ready_datasets/files/RCM_1961-1980.zarr      # Predictands
```

______________________________________________________________________

## 6. Train CPMGEM-residual — a two-step workflow

### 6.1. Step 1 — Train the deterministic SongUNet regressor (`sqrt + minmax_neg1_1`)

The regressor is a SongUNet trained *deterministically* (noisy slot = 0, noise label `t` = 0, conditioning only through `cond_low`) to predict the conditional mean `ŷ`. The residual stage then models `r = y − ŷ` **in the regressor's normalized space**, so the regressor's predictand transform fixes the space the whole cell lives in.

> ⚠️ **Use the 2×2 transform here, not ResDiff's.** This regressor is structurally the same as the [ResDiff regressor](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20ResDiff/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20ResDiff.md#61-step-1--train-the-deterministic-songunet-regressor), but the predictand normalizer differs: here it is `sqrt + minmax_neg1_1` (the clean-2×2 convention), whereas the ResDiff tutorial uses `std`. So CPMGEM-residual needs **its own regressor checkpoint** — do not reuse ResDiff's `std`-trained one, or the residuals would be computed in the wrong space.

`./training/configs/song_unet_det_minmax.yaml`:

```yaml
##### GENERAL INFO #####
run_ID: song_unet_det_minmax
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
    # 2x2 convention: sqrt then minmax_neg1_1 (NOT std). Must match the residual
    # stage's predictand transform in section 6.2.
    operator:
      default: sqrt
    normalizer:
      path_reference: ./AI_ready_datasets/files/RCM_1961-1980.zarr
      default: minmax_neg1_1
    transform_to_2D: true


##### DATA LOADER CONFIGURATION #####
dataloader:
  batch_size: 16
  shuffle: true
  num_workers: 1


##### MODEL CONFIGURATION #####
model_info:
  saving_params:
    model_save_name: SongUNet_det_minmax
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
      in_channels: 1
      cond_low_channels: 15
      cond_high_channels: 0
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
      spatial_pe_freqs: 1

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
d4p-train ./training/configs/song_unet_det_minmax.yaml
```

The best checkpoint is written to `./outputs/song_unet_det_minmax/models/SongUNet_det_minmax_best.pt`. **You will need this path for step 2.**

### 6.2. Step 2 — Train the sub-VP residual diffusion model

This stage reuses ResDiff's residual machinery but swaps the diffusion engine:

1. **`d4p_pydataset`** — the **same** `pydataset_resdiff` as ResDiff. It runs the regressor on every date, caches `r = y − ŷ` and the regression mean `ŷ` to a Zarr, and serves `(r, c_low, c_high)` triples (`c_high` is `ŷ`).

1. **`d4p_trainer`** — `trainer_cpmgem_residual`. It inherits ResDiff's data/regressor wiring but runs the **continuous-time sub-VP SDE** forward process on the residual and predicts the noise ε with a **plain SongUNet** (no EDM preconditioner), trained with MSE on ε.

1. **Residual standardization (`standardize_residual: true`)** — the one genuinely new ingredient. The sub-VP marginal (`mean(t)=e^{−½B(t)}`, `std(t)=1−e^{−B(t)}`, prior `N(0,I)`) assumes the clean signal has ~unit variance. CPMGEM-direct gets this for free because its target `y` is min-max rescaled to `[−1,1]`. The **residual** of normalized fields has std ≪ 1, and — unlike EDM, which absorbs the data scale through its `sigma_data` preconditioning — sub-VP has no scale-handling mechanism. So the trainer standardizes the residual to unit variance (per-variable, using the mean/std already stored in the residuals Zarr), runs sub-VP in standardized space, and saves those stats to the checkpoint metadata so the downscaler can invert them.

1. **High-res conditioning (`cond_high_channels: 1`)** — `ŷ` is fed back to the U-Net as a high-res conditioning channel, exactly as in ResDiff.

`./training/configs/cpmgem_residual.yaml`:

```yaml
##### GENERAL INFO #####
run_ID: cpmgem_residual
output_dir: ./outputs
overwrite: true


##### TRAINER SELECTION #####
# Sub-VP SDE diffusion on the regression residual (sub-VP × residual cell).
d4p_trainer:
  name: trainer_custom
  module: deep4production.core.trainers.trainer_cpmgem_residual


##### CUSTOM PYDATASET (same machinery as ResDiff) #####
d4p_pydataset:
  name: pydataset_custom
  module: deep4production.core.pydatasets.pydataset_resdiff
  kwargs:
    # The sqrt+minmax regressor from step 6.1 (NOT ResDiff's std-trained one).
    path_regressor: ./outputs/song_unet_det_minmax/models/SongUNet_det_minmax_best.pt
    add_pred_mean: true        # feed ŷ as cond_high
    add_context_lowres: true   # feed raw predictors as cond_low
    residuals:
      path: ./outputs/cpmgem_residual/aux_files/residuals.zarr
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
    # Must match the regressor's transform from step 6.1 (sqrt + minmax_neg1_1),
    # so the residual r = y - ŷ is well-defined. Same convention as CPMGEM and
    # EDM-direct; ResDiff is the only cell on a different (std) normalization.
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
    model_save_name: CPMGEM_residual

  # MSE on predicted noise ε (sub-VP formulation — no EDM λ(σ) weighting).
  loss_params:
    name: MseLoss
    module: deep4production.deep.loss
    kwargs:
      ignore_nans: false

  # PLAIN SongUNet (no EDM preconditioner — sub-VP calls the backbone directly).
  # cond_high carries the regressor mean ŷ, so cond_high_channels = C_y.
  model_params:
    name: SongUNet
    module: deep4production.deep.models.unet.song_unet
    kwargs:
      in_channels: 1            # C_y — channels of the residual
      cond_low_channels: 15     # C_x — predictor channels (d4p-inspect)
      cond_high_channels: 1     # C_yhat — regression mean as high-res context
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

    # sub-VP SDE noise schedule + residual standardization.
    kwargs:
      noise_params:
        beta_min: 0.1
        beta_max: 20.0
        t_min: 1.0e-5
      standardize_residual: true   # essential for sub-VP (see section 6.2)
```

Train it:

```bash
d4p-train ./training/configs/cpmgem_residual.yaml
```

> 💡 **First-run cost.** Like ResDiff, the very first run iterates over every date, runs the regressor, and writes `residuals_training.zarr` + `residuals_validation.zarr`. Subsequent runs reuse the cache. Note these residuals are specific to *this* regressor (`sqrt + minmax`), so they are a **different** cache from ResDiff's `std` residuals — keep `residuals.path` distinct.

______________________________________________________________________

## 7. Run Inference with `d4p-downscale`

The CPMGEM-residual downscaler:

1. Loads the regressor from `path_regressor` (stored in the training metadata).
1. For each date: runs `regressor(x) → ŷ`, then samples a **standardized residual** with the **reverse sub-VP SDE** chain (Euler–Maruyama), conditioned on `(x, ŷ)`.
1. Un-standardizes the residual (`r̂ = r_std·σ_res + μ_res`, using the stats saved at training) and returns `ŷ + r̂`.

`./inference/configs/cpmgem_residual.yaml`:

```yaml
##### GENERAL INFO #####
id_dir: ./outputs/cpmgem_residual


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
# Plain SongUNet (no EDM preconditioner). Metadata also stores path_regressor and
# residual_norm (the per-variable residual mean/std), consumed automatically.
model_file: CPMGEM_residual_best.pt


##### DOWNSCALER CLASS #####
d4p_downscaler:
  name: downscaler_custom
  module: deep4production.core.downscalers.downscaler_cpmgem_residual
  kwargs:
    # Optional: override the regressor path stored in training metadata.
    # path_regressor: ./outputs/song_unet_det_minmax/models/SongUNet_det_minmax_best.pt
    sampling_params:
      # Reverse sub-VP SDE steps (as in CPMGEM). 200–500 is a good trade-off.
      num_steps: 1000
      denoise: true
      # beta_min / beta_max read automatically from training metadata.
      # t_min decoupled from training (default 1e-3) to avoid the σ→0 blow-up.
      # t_min: 1.0e-3


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
d4p-downscale ./inference/configs/cpmgem_residual.yaml
```

> ⚖️ **Cost.** Like CPMGEM (and unlike ResDiff), the sub-VP reverse SDE uses many steps (~1000), so CPMGEM-residual is markedly slower at inference than ResDiff's ~18-step EDM Heun sampler — even though both produce a residual. This step-count difference is itself one of the axes the 2×2 study controls for. For quick smoke tests use `num_steps: 50` and `ensemble_size: 1`.

The output NetCDF has shape `(member, time, point)`.

______________________________________________________________________

## 8. Visualization

Identical to the CPMGEM/ResDiff tutorials — point `plot_date_from_1D_spatial_field` at `./outputs/cpmgem_residual/predictions/1980.nc` and pick a member:

```python
import xarray as xr
from deep4production.viz.spatial import plot_date_from_1D_spatial_field

kwargs = {
    "date": "1980-01-01",
    "vmin": 0, "vmax": 10,
    "set_extent": [5, 15, 44, 48],
    "central_longitude": 0,
    "cbar_label": "Precipitation (mm)",
    "titles": ["Groundtruth (RCM)", "Prediction (CPMGEM-residual, member 0)", "Difference"],
    "diff": True, "vminDiff": -5, "vmaxDiff": 5, "cmapDiff": "BrBG",
}

tgt = xr.open_dataset(
    "./source_files/data_zenodo/ALPS_domain/train/ESD_pseudo_reality/target/"
    "pr_tasmax_CNRM-CM5_1961-1980.nc"
).stack(point=("y", "x"))
tgt['time'] = tgt.time.dt.floor('D')

prd = xr.open_dataset("./outputs/cpmgem_residual/predictions/1980.nc").isel(member=0)
prd['time'] = prd.time.dt.floor('D')

kwargs.update({"data": [tgt["pr"], prd["pr"]]})
fig = plot_date_from_1D_spatial_field(**kwargs)
```

As with ResDiff, you can decompose the prediction into the regressor mean `ŷ` and the diffusion residual `r̂ = prediction − ŷ` to see which stage contributes what.

______________________________________________________________________

## 9. Summary

You have run the **sub-VP × residual** cell of the 2×2 study:

- Trained a deterministic SongUNet regressor on the 2×2 `sqrt + minmax_neg1_1` transform (step 1) — its own checkpoint, not ResDiff's `std` one.
- Trained a **plain** SongUNet to denoise the **residual** under the **continuous-time sub-VP SDE**, with the residual standardized to unit variance so the sub-VP schedule is well-conditioned.
- Generated ensemble fields with the regressor + reverse sub-VP sampler, un-standardizing the residual before adding it back to `ŷ`.

Compare against [CPMGEM](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20CPMGEM/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20CPMGEM.md) (same sub-VP formulation, direct target — and the same `sqrt + minmax` convention, so the target axis is cleanly isolated). The EDM-residual corner is represented by [ResDiff](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20ResDiff/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20ResDiff.md), but on CorrDiff's native `std` normalization — so a CPMGEM-residual ↔ ResDiff comparison mixes the formulation axis with the normalization difference. To isolate the formulation axis cleanly you would train a `sqrt + minmax` variant of the EDM-residual cell.

______________________________________________________________________

## 10. References

- [CORDEX-BENCH GitHub](https://github.com/WCRP-CORDEX/ml-benchmark) · [Zenodo](https://zenodo.org/records/17957264)
- Addison et al. 2024 — *Machine learning emulation of precipitation … using a diffusion model* (CPMGEM, sub-VP)
- Mardani et al. 2023 — *Generative Residual Diffusion Modeling for Km-Scale Atmospheric Downscaling* (CorrDiff / residual)
- Song et al. 2021 — *Score-based generative modeling through SDEs* (NCSN++ / sub-VP SDE)
