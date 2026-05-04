# Recipes

This directory contains YAML configuration templates for all **deep4production** workflows. Recipes are passed directly to the CLI commands and drive the entire pipeline without touching framework code.

---

## Creating AI-ready datasets

Before training, raw NetCDF files must be converted to the d4p Zarr format using `d4p-create`. The template in `create_datasets/template.yaml` shows how to specify input paths, variables, temporal range, and NaN imputation strategy. Run `d4p-inspect` on the resulting Zarr to verify its structure.

---

## Training & inference recipes

Each entry below represents one model configuration. Where both training and inference recipes exist, they share the same name across `training/` and `inference/`.

### DeepESD — MSE loss
**Files:** `training/deepesd_mse.yaml` · `inference/standard.yaml`

Training and inference recipes for a DeepESD CNN trained with mean squared error loss. The simplest and fastest baseline for deterministic downscaling. [[1]](#ref-1)

---

### DeepESD — Asymmetric loss
**Files:** `training/deepesd_asym.yaml` · `inference/standard.yaml`

Training and inference recipes for a DeepESD CNN trained with an asymmetric Gamma-based loss that penalises underestimation of extreme values more heavily than overestimation. Recommended for precipitation. [[1]](#ref-1)

---

### DeepESD — NLL BerGamma loss
**Files:** `training/deepesd_bg.yaml` · `inference/standard.yaml`

Training and inference recipes for a DeepESD CNN trained with a Bernoulli–Gamma negative log-likelihood loss, producing probabilistic predictions for mixed discrete–continuous variables such as precipitation. [[1]](#ref-1)

---

### SongUNet — Deterministic, asymmetric loss
**Files:** `training/song_unet_det_asym.yaml` · `inference/song_unet_det_asym.yaml`

Training and inference recipes for a deterministic SongUNet (NCSN++ backbone) with asymmetric loss. The diffusion pathway is bypassed; the U-Net is conditioned exclusively on the low-resolution predictor field. [[2]](#ref-2)

---

### GNN4CD — Quantised MSE loss
**Files:** `training/gnn4cd_qmse.yaml` · `inference/gnn4cd_qmse.yaml`

Training and inference recipes for a graph neural network operating on a bipartite heterogeneous graph between low- and high-resolution grid nodes. Suited for irregular or unstructured grids. The graph must be pre-built once from the Zarr files using the `build_graph` helper and referenced in the recipe. [[3]](#ref-3)

---

### ResDiff — Residual diffusion (CorrDiff-style)
**Files:** `training/resdiff.yaml` · `inference/resdiff.yaml`

Training and inference recipes for a residual diffusion model. A separately trained deterministic regressor provides a mean prediction; the diffusion model (EDM preconditioner + SongUNet) learns the residual distribution. Requires a pre-trained regressor checkpoint. [[2]](#ref-2) [[4]](#ref-4)

---

### CPMGEM — Continuous-time sub-VP SDE diffusion
**Files:** `training/cpmgem.yaml` · `inference/cpmgem.yaml`

Training and inference recipes for a direct-generation diffusion model based on a continuous-time sub-variance-preserving SDE. Generates high-resolution fields in a single end-to-end diffusion process without a separate regressor. [[2]](#ref-2)

---

## References

<a id="ref-1"></a>[1] Baño-Medina, J., Manzanas, R., Cimadevilla, E., Fernández, J., González-Abad, J., Cofiño, A. S., & Gutiérrez, J. M. (2022). Downscaling multi-model climate projection ensembles with deep learning (DeepESD): contribution to CORDEX EUR-44. *Geoscientific Model Development Discussions*, 2022, 1–14.

<a id="ref-2"></a>[2] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2020). Score-based generative modeling through stochastic differential equations. *arXiv preprint arXiv:2011.13456*.

<a id="ref-3"></a>[3] Blasone, V., Coppola, E., Sanguinetti, G., Arora, V., Di Gioia, S., & Bortolussi, L. (2025). Graph neural networks for hourly precipitation projections at the convection permitting scale with a novel hybrid imperfect framework. *Environmental Data Science*, 4, e47.

<a id="ref-4"></a>[4] Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the design space of diffusion-based generative models. *Advances in Neural Information Processing Systems*, 35, 26565–26577.
