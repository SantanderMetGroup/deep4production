# Recipes

This directory contains YAML configuration templates for all **deep4production** workflows. Recipes are passed directly to the CLI commands and drive the entire pipeline without touching framework code.

______________________________________________________________________

## Run-directory convention

Every run lives in **one self-contained directory** named after its `run_ID`. That directory (`id_dir = output_dir/run_ID`) holds the run's recipes, and everything the run *generates* goes under its `outputs/` subdirectory:

```
<output_dir>/
  <run_ID>/                 # = id_dir — the run directory
    train.yaml              # d4p-train recipe
    inference.yaml          # d4p-downscale recipe
    outputs/                # everything the run generates
      models/               #   checkpoints (.pt, metadata embedded)
      aux_files/            #   caches (residuals, graphs, Gamma params, ...)
      predictions/          #   downscaled NetCDF output
      tracker/              #   d4p-tracker figures + CSV
```

Both `run_ID` and `output_dir` are **required** in every `train.yaml` and `inference.yaml` — there is no default and no standalone `id_dir` key. Set `run_ID` to the directory's name and `output_dir` to its parent (typically your project root), e.g. `run_ID: RESDIFF` and `output_dir: /gpfs/.../projects/TROPICAL-BENCH` resolve to `.../TROPICAL-BENCH/RESDIFF/`. Inference reuses the same `run_ID` + `output_dir` as training, so `model_file` resolves under `id_dir/outputs/models/` and predictions are written to `id_dir/outputs/predictions/`.

**A second configuration = a second directory.** To try a variant, create another sibling directory with its own `run_ID` and its own `train.yaml` / `inference.yaml` (e.g. `RESDIFF_v2/`).

Each model directory below is a ready-to-copy template. Copy `recipes/<MODEL>/` into your project, edit the paths, and run `d4p-train ./train.yaml` / `d4p-downscale ./inference.yaml` from inside it.

______________________________________________________________________

## Creating AI-ready datasets

Before training, raw NetCDF files must be converted to the d4p Zarr format using `d4p-create`. The template in `create_datasets/template.yaml` shows how to specify input paths, variables, temporal range, and NaN imputation strategy. Run `d4p-inspect` on the resulting Zarr to verify its structure.

______________________________________________________________________

## Model recipe directories

Each entry below is one model directory containing `train.yaml` and `inference.yaml`.

### DeepESD — MSE loss

**Files:** `DEEPESD_MSE/train.yaml` · `DEEPESD_MSE/inference.yaml`

Training and inference recipes for a DeepESD CNN trained with mean squared error loss. The simplest and fastest baseline for deterministic downscaling. [\[1\]](#ref-1)

______________________________________________________________________

### DeepESD — Asymmetric loss

**Files:** `DEEPESD_ASYM/train.yaml` · `DEEPESD_ASYM/inference.yaml`

Training and inference recipes for a DeepESD CNN trained with an asymmetric Gamma-based loss that penalises underestimation of extreme values more heavily than overestimation. Recommended for precipitation. [\[1\]](#ref-1)

______________________________________________________________________

### DeepESD — NLL BerGamma loss

**Files:** `DEEPESD_BG/train.yaml` · `DEEPESD_BG/inference.yaml`

Training and inference recipes for a DeepESD CNN trained with a Bernoulli–Gamma negative log-likelihood loss, producing probabilistic predictions for mixed discrete–continuous variables such as precipitation. [\[1\]](#ref-1)

______________________________________________________________________

### SongUNet — Deterministic, asymmetric loss

**Files:** `SONG_UNET_DET_ASYM/train.yaml` · `SONG_UNET_DET_ASYM/inference.yaml`

Training and inference recipes for a deterministic SongUNet (NCSN++ backbone) with asymmetric loss. The diffusion pathway is bypassed; the U-Net is conditioned exclusively on the low-resolution predictor field. [\[2\]](#ref-2)

______________________________________________________________________

### GNN4CD — Quantised MSE loss

**Files:** `GNN4CD/train.yaml` · `GNN4CD/inference.yaml`

Training and inference recipes for a graph neural network operating on a bipartite heterogeneous graph between low- and high-resolution grid nodes. Suited for irregular or unstructured grids. The graph must be pre-built once from the Zarr files using the `build_graph` helper and referenced in the recipe. [\[3\]](#ref-3)

______________________________________________________________________

### ResDiff — Residual diffusion (CorrDiff-inspired)

**Files:** `RESDIFF/train.yaml` · `RESDIFF/inference.yaml`

Training and inference recipes for a residual diffusion model inspired by CorrDiff from NVIDIA [\[5\]](#ref-5), though not an exact reimplementation. A separately trained deterministic regressor provides a mean prediction; the diffusion model (EDM preconditioner + SongUNet) learns the residual distribution. Requires a pre-trained regressor checkpoint. [\[2\]](#ref-2) [\[4\]](#ref-4)

______________________________________________________________________

### CPMGEM — Continuous-time sub-VP SDE diffusion

**Files:** `CPMGEM/train.yaml` · `CPMGEM/inference.yaml`

Training and inference recipes for a direct-generation diffusion model based on a continuous-time sub-variance-preserving SDE. Generates high-resolution fields in a single end-to-end diffusion process without a separate regressor. [\[2\]](#ref-2) [\[6\]](#ref-6)

______________________________________________________________________

## Explainability (XAI) recipes

Gradient input-attribution with `d4p-explain`. These recipes compute the
gradient of a reduced predictand response (a chosen `target_var`, reduced over a
predictand box or gridpoint) with respect to the predictor input, and write
per-date saliency maps plus a per-channel contribution ranking to
`id_dir/outputs/xai/`. Useful to diagnose *which predictor channels a model
relies on* — e.g. comparing a univariate vs a multivariate emulator, or perfect
(UPSRCM) vs imperfect (GCM) predictors. Valid for **deterministic regressors**
(MSE/asym DeepESD, SongUNet); distributional-loss checkpoints (BerGamma/Gaussian)
are refused.

An `explain.yaml` lives alongside `train.yaml` /
`inference.yaml` inside the model's run directory, reusing the same
`run_ID` + `output_dir`, so `d4p-explain ./explain.yaml` reads the checkpoint
from `id_dir/outputs/models/` and writes maps to `id_dir/outputs/xai/`.

### DeepESD — gradient attribution

**File:** `DEEPESD_MSE/explain.yaml`

Base `Explainer` (forward `model(x, f)`) for a deterministic DeepESD CNN. [\[1\]](#ref-1)

______________________________________________________________________

### SongUNet (deterministic) — gradient attribution

**File:** `SONG_UNET_DET_ASYM/explain.yaml`

`ExplainerSongUNetDet` (forward conditions on `cond_low`) for the deterministic
SongUNet baselines, including the multivariate variant. [\[2\]](#ref-2)

______________________________________________________________________

## References

<a id="ref-1"></a>\[1\] Baño-Medina, J., Manzanas, R., Cimadevilla, E., Fernández, J., González-Abad, J., Cofiño, A. S., & Gutiérrez, J. M. (2022). Downscaling multi-model climate projection ensembles with deep learning (DeepESD): contribution to CORDEX EUR-44. *Geoscientific Model Development Discussions*, 2022, 1–14.

<a id="ref-2"></a>\[2\] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2020). Score-based generative modeling through stochastic differential equations. *arXiv preprint arXiv:2011.13456*.

<a id="ref-3"></a>\[3\] Blasone, V., Coppola, E., Sanguinetti, G., Arora, V., Di Gioia, S., & Bortolussi, L. (2025). Graph neural networks for hourly precipitation projections at the convection permitting scale with a novel hybrid imperfect framework. *Environmental Data Science*, 4, e47.

<a id="ref-4"></a>\[4\] Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the design space of diffusion-based generative models. *Advances in Neural Information Processing Systems*, 35, 26565–26577.

<a id="ref-5"></a>\[5\] Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C. Y., Liu, C. C., ... & Pritchard, M. (2025). Residual corrective diffusion modeling for km-scale atmospheric downscaling. *Communications Earth & Environment*, 6(1), 124.

<a id="ref-6"></a>\[6\] Addison, H., Kendon, E. J., Ravuri, S., Aitchison, L., & Watson, P. A. (2026). Machine learning emulation of precipitation from km‐scale UK regional climate simulations using a diffusion model. *Journal of Advances in Modeling Earth Systems*, 18(3), e2025MS005140.
