# Deep4Production Tutorial: CORDEX-BENCH Alps Case Study with CPMGEM

This tutorial demonstrates how to use the [deep4production](https://github.com/SantanderMetGroup/deep4production) framework for **probabilistic** climate downscaling using **CPMGEM** (Convection-Permitting Model — Generative Emulation Method, Addison et al. 2024). CPMGEM is a continuous-time **sub-VP SDE** diffusion model that directly generates high-resolution precipitation conditioned on coarse predictors. We use the same CORDEX-BENCH Alps domain as in the DeepESD tutorial, so we can focus on the parts that differ for diffusion models.

> If you are new to `deep4production`, please go through the [DeepESD tutorial](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md) first — sections 1 through 5 (project structure, data download, dataset creation, inspection) are identical and are summarized briefly here.

---

## 1. Introduction

**deep4production** is a modular CLI for deep-learning climate downscaling. The four CLI tools (`d4p-create`, `d4p-inspect`, `d4p-train`, `d4p-downscale`) and the project structure are described in the [DeepESD tutorial, section 1](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md#1-introduction).

This tutorial swaps the deterministic **DeepESD** backbone for a **stochastic diffusion** model, so each forward pass is now a *reverse SDE trajectory* and a single model can produce multiple plausible realizations of the high-resolution field per input.

---

## 2. Case study: CORDEX-BENCH

Same simplified CORDEX-BENCH Alps configuration as in the DeepESD tutorial:

* **Domain:** Central Europe (Alps)
* **AI-model backbone:** **CPMGEM** (sub-VP SDE diffusion, NCSN++/SongUNet UNet)
* **Loss function:** MSE on predicted noise ε
* **Predictors:** UPSRCM (16 × 16, 15 variables: `u/v/t/q/z` at 850/700/500 hPa)
* **Predictands:** RCM (128 × 128) precipitation `pr`
* **Training:** 1961-1979 (excl. 1967, 1975) · **Validation:** 1967, 1975 · **Test:** 1980

---

## 3. Download CORDEX-BENCH Alps Data

Identical to the DeepESD tutorial — see [section 3 there](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md#3-download-cordex-bench-alps-data).

---

## 4. Prepare AI-Ready Datasets with `d4p-create`

Identical to the DeepESD tutorial. The CPMGEM trainer reads from the **same Zarr files** (`UPSRCM_1961-1980.zarr` and `RCM_1961-1980.zarr`) — no data-side changes are required when switching architectures. See [section 4 of the DeepESD tutorial](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md#4-prepare-ai-ready-datasets-with-d4p-create).

---

## 5. Inspect the Zarr Datasets with `d4p-inspect`

```bash
d4p-inspect ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr   # Predictors
d4p-inspect ./AI_ready_datasets/files/RCM_1961-1980.zarr      # Predictands
```

Use the reported shape `(C, H, W)` to fill in the `cond_low_channels`, `in_channels`, and the implicit grid of the diffusion U-Net (see section 6).

---

## 6. Train CPMGEM with `d4p-train`

CPMGEM differs from DeepESD in three places:

1. **Custom trainer (`d4p_trainer`).** Diffusion models need a noise schedule and an integration over noise levels at training time. The DeepESD recipe doesn't declare a `d4p_trainer` block — it uses the default deterministic trainer. CPMGEM must point to `trainer_cpmgem`, which implements the sub-VP SDE training loop.

2. **Predictand transform (`operator` + `normalizer`).** Precipitation is heavy-tailed and zero-inflated; CPMGEM trains on a transformed target so the diffusion process operates in a roughly Gaussian space:
   * `operator: sqrt` — applied first, compresses the upper tail.
   * `normalizer: minmax_neg1_1` — rescales to `[-1, 1]` (the natural range of the noise the SDE adds). Statistics are derived from the raw zarr stats; no recomputation needed.

3. **Architecture and training schedule.** A larger NCSN++/SongUNet replaces the small DeepESD CNN. Diffusion training also benefits from EMA weight averaging, a warm-up scheduler, and gradient clipping — all enabled via the YAML.

Below is the full CPMGEM training YAML (`./training/configs/cpmgem.yaml`):

```yaml
##### GENERAL INFO #####
run_ID: cpmgem
output_dir: ./outputs
overwrite: true

# Continuous-time sub-VP SDE diffusion trainer (direct generation, no residual).
d4p_trainer:
  name: trainer_custom
  module: deep4production.core.trainers.trainer_cpmgem


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
    # Paper transform: pr → sqrt(pr) → linearly rescale to [-1, 1].
    # The operator is applied first, then the normalizer. Because sqrt is
    # monotone on pr ≥ 0, the post-operator min/max are derived automatically
    # from the raw zarr stats (sqrt(min_raw), sqrt(max_raw)); no need to
    # recompute the zarr.
    operator:
      default: sqrt
    normalizer:
      path_reference: ./AI_ready_datasets/files/RCM_1961-1980.zarr
      default: minmax_neg1_1   # 2*(x-min)/(max-min) - 1
    transform_to_2D: True


##### DATA LOADER CONFIGURATION #####
dataloader:
  batch_size: 16
  shuffle: true
  num_workers: 4


##### MODEL CONFIGURATION #####
model_info:
  saving_params:
    model_save_name: CPMGEM

  # MSE on predicted noise ε. No NaNs in noise → ignore_nans: false.
  loss_params:
    name: MseLoss
    module: deep4production.deep.loss
    kwargs:
      ignore_nans: false

  # Conditioned NCSN++ U-Net (SongUNet backbone) — only the low-res
  # conditioning stream is enabled (cond_high_channels: 0).
  model_params:
    name: SongUNet
    module: deep4production.deep.models.unet.song_unet
    kwargs:
      in_channels: 1              # C_y — single precipitation field
      cond_low_channels: 15       # C_x — predictor channels (d4p-inspect)
      cond_high_channels: 0       # CPMGEM does not use high-res conditioning
      nf: 128                     # base channel width
      ch_mult: [1, 2, 2, 2]       # 4 encoder levels
      num_res_blocks: 4
      attn_at_levels: [3]
      dropout: 0.1
      fir: true
      fir_kernel: [1, 3, 3, 1]
      skip_rescale: true
      progressive_input: true
      cond_upsample: nearest      # paper-exact: nearest-neighbour up-sampling

  training_params:
    num_epochs: 20
    patience_early_stopping: 5
    optimizer_params:
      lr: 0.0002                  # Adam lr (paper value)
    ema_decay: 0.9999             # EMA for weight stabilisation
    grad_clip: 1.0                # L2-norm gradient clipping

    # Linear warm-up over 5000 steps then constant LR.
    scheduler_params:
      type: LambdaLR
      lr_lambda: rampup_expdecay
      kwargs:
        base_lr: 0.0002
        rampup_steps: 5000
        decay_rate: 1.0           # 1.0 → no decay (constant LR after warm-up)
        decay_interval: 1
        terminal_value: 1.0e-6

    # sub-VP SDE noise schedule (Song et al. 2021).
    kwargs:
      noise_params:
        beta_min: 0.1
        beta_max: 20.0
        t_min: 1.0e-5
```

Train with:

```bash
d4p-train ./training/configs/cpmgem.yaml
```

> 💡 **Tip — first epoch is slow.** The trainer compiles internal schedulers and (optionally) torch-compiles the model on the first forward pass. From epoch 2 onwards throughput is much higher.

Below is an example of training output:

![d4p-train-1](./images/d4p-train-output.png)

---

### Enabling MLflow

The MLflow block is identical to the [DeepESD tutorial section](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md#enabling-mlflow-in-deep4production); just change the `model: cpmgem` and `loss: mse` tags so runs are easy to filter in the MLflow UI.

A small but very useful tweak for diffusion runs: set `compute_diagnostics_every_n_epochs` higher than for DeepESD (e.g. `5`) — generating diagnostic samples requires a full reverse SDE rollout per sample, which is far more expensive than a single forward pass.

---

## 7. Run Inference with `d4p-downscale`

The inference YAML is short — the heavy lifting (model architecture, noise schedule) is reconstructed automatically from the metadata stored inside the checkpoint by `d4p-train`.

```yaml
##### GENERAL INFO #####
# Directory created by d4p-train for the cpmgem run_ID.
id_dir: ./outputs/cpmgem


##### INPUT DATA #####
input_data:
  paths:
    - ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr
  years: [1980]
  load_in_memory: true


##### ENSEMBLE SIZE #####
# Each reverse-diffusion trajectory draws fresh Gaussian noise, so
# ensemble_size > 1 produces independent realisations — useful for
# uncertainty estimation.
ensemble_size: 5


##### MODEL CHECKPOINT #####
model_file: CPMGEM_best.pt   # relative to id_dir/models/


##### DOWNSCALER CLASS #####
# CPMGEM-specific subclass: replaces the deterministic forward pass with a
# reverse sub-VP SDE sampling chain.
d4p_downscaler:
  name: downscaler_custom
  module: deep4production.core.downscalers.downscaler_cpmgem
  kwargs:
    sampling_params:
      # Number of reverse SDE steps.
      # Paper value: 1000.  200–500 gives a good quality/speed trade-off.
      num_steps: 1000
      # Deterministic (noise-free) final step — reduces residual variance and
      # produces slightly sharper outputs.
      denoise: true
      # beta_min / beta_max / t_min are read automatically from the checkpoint
      # metadata. Override here only if needed.
      # beta_min: 0.1
      # beta_max: 20.0
      # t_min: 1.0e-5


##### OUTPUT #####
saving_info:
  file: 1980.nc
  template: null
  formatting: null


##### INFERENCE RUNTIME #####
# Forwarded as **kwargs to downscaler.downscale().
inference_params:
  batch_size: 4         # number of dates whose reverse-SDE chains run in parallel on the GPU
  amp_dtype: null       # 'bfloat16' (Ampere+) / 'float16' / null
  compile: false        # torch.compile the score network — amortises well over 1000 SDE steps
```

Run with:

```bash
d4p-downscale ./inference/configs/cpmgem.yaml
```

Below is an example of inference output:

![d4p-downscale-1](./images/d4p-downscale-output.png)

> ⚠️ **Cost note.** Inference cost is approximately `ensemble_size × num_steps × cost(forward UNet)`. With the paper defaults (5 members × 1000 steps) one year takes substantially longer than with DeepESD. For quick smoke tests, set `num_steps: 50` and `ensemble_size: 1`.

The output NetCDF has shape `(member, time, point)` — one extra dimension compared to deterministic models. Aggregating across `member` gives the ensemble mean; the spread captures forecast uncertainty:

![d4p-downscale-2](./images/d4p-downscale-pred.png)

---

## 8. Visualization

The same `plot_date_from_1D_spatial_field` helper used in the DeepESD tutorial works here — just point it at the CPMGEM prediction file and select an ensemble member:

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
    "titles": ["Groundtruth (RCM)", "Prediction (CPMGEM, member 0)", "Difference"],
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

prd = xr.open_dataset("./outputs/cpmgem/predictions/1980.nc")
prd = prd.isel(member=0)              # pick a single realisation
prd['time'] = prd.time.dt.floor('D')

var = "pr"
kwargs.update({"data": [tgt[var], prd[var]]})

fig = plot_date_from_1D_spatial_field(**kwargs)
```

![figure](./images/cpmgem_1980-01-01.png)

### Going beyond a single member

Because CPMGEM is *probabilistic*, a single map only shows one of many plausible realizations. Two common follow-ups:

```python
# Ensemble mean — typically the best deterministic estimator under MSE.
prd_mean = prd.mean(dim="member")

# Ensemble spread — proxy for predictive uncertainty.
prd_std  = prd.std(dim="member")
```

---

## 9. Summary

You have now run a full diffusion-based downscaling workflow on CORDEX-BENCH Alps using CPMGEM:

* Reused the AI-ready datasets created in the DeepESD tutorial.
* Trained an NCSN++/SongUNet via continuous-time sub-VP SDE noise denoising.
* Generated **ensemble** high-resolution precipitation fields by running multiple reverse SDE trajectories.
* Visualized one member and outlined the standard ensemble-mean / spread post-processing.

For a residual diffusion variant (separately trained regressor + EDM-preconditioned diffusion of the residual), see the [ResDiff tutorial](../Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20ResDiff/Deep4Production%20Tutorial:%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20ResDiff.md).

---

## 10. References

* [CORDEX-BENCH GitHub](https://github.com/WCRP-CORDEX/ml-benchmark)
* [CORDEX-BENCH Zenodo](https://zenodo.org/records/17957264)
* Addison et al. 2024 — *Machine learning emulation of precipitation from km-scale regional climate simulations using a diffusion model*
* Song et al. 2021 — *Score-based generative modeling through SDEs* (NCSN++ / sub-VP SDE)
