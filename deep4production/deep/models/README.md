# Model Architectures

This package contains all neural network architectures available in **deep4production**, organized by family. Models are referenced in YAML recipes via their `module` and `name` fields, which are resolved dynamically at runtime.

---

## Available architectures

### Convolutional Neural Networks (`cnn/`)

Lightweight, fast-to-train CNN models suitable for deterministic downscaling.

| Class | File | Description |
|---|---|---|
| `DeepESD` | `cnn/DeepESD.py` | Three-layer convolutional network for spatial downscaling. Configurable filter sizes, kernel size, and output activation. Supports ensemble-style multi-output heads when used with the DeepESD trainer. [[1]](#ref-1) |
| `SMHICNN` | `cnn/smhi_cnn.py` | CNN with dilated convolutions and pixel-shuffle upscaling, following the SMHI design. Supports learnable pre-maps, multiple dilation rates, and configurable pixel-shuffle blocks. [[2]](#ref-2) |

---

### U-Net architectures (`unet/`)

Encoder-decoder networks with skip connections, suited for high-resolution spatial output.

| Class | File | Description |
|---|---|---|
| `SongUNet` | `unet/song_unet.py` | Generalized NCSN++ U-Net backbone. Accepts a noisy input `x`, a scalar noise label `t`, and optional low-res and high-res conditioning streams. Used as the backbone inside `EDMPrecond` for diffusion models, or standalone as a deterministic downscaler. Supports FIR-filtered up/downsampling, attention at configurable levels, and spatial positional embeddings. [[3]](#ref-3) |
| `abad_unet` | `unet/abad_unet.py` | UNet-style encoder-decoder with transposed convolution or bilinear upsampling. Configurable base channels, batch normalisation, and output activation. Designed for deterministic regression on spatial grids. [[4]](#ref-4) |

---

### Graph Neural Networks (`gnn/`)

Point-based models that operate directly on irregular grids, without assuming a regular spatial layout.

| Class / function | File | Description |
|---|---|---|
| `GNN4CD` | `gnn/GNN4CD.py` | Graph neural network for climate downscaling. Operates on a bipartite heterogeneous graph between low-resolution and high-resolution nodes. Uses RNN-based temporal encoding and message-passing graph operations. [[5]](#ref-5) |
| `build_graph` | `gnn/GNN4CD.py` | Helper that constructs the PyG `HeteroData` graph from two Zarr files (low-res and high-res). Connects each high node to its `k` nearest low nodes and builds bidirectional high–high edges. Must be called once before training and the result passed to the trainer via the recipe. |

---

### Diffusion / score-based models (`diffusion/`)

Wrappers that turn a backbone U-Net into an EDM-compatible denoising model.

| Class | File | Description |
|---|---|---|
| `EDMPrecond` | `diffusion/edm_precond.py` | EDM preconditioner wrapping any backbone `F_θ`. Applies the `c_skip`, `c_out`, `c_in`, `c_noise` scalings so the backbone sees a well-conditioned denoising problem. Used with `SongUNet` for the ResDiff / CorrDiff-style trainers. [[6]](#ref-6) |

---

## Recipe usage

Reference a model by class name and module path under `model_info.model_params`:

```yaml
model_info:
  model_params:
    name: DeepESD
    module: deep4production.deep.models.cnn.DeepESD
    kwargs:
      x_shape: [15, 16, 16]   # (C, H, W) — from d4p-inspect
      y_shape: [1, 128, 128]
      filters: [50, 25, 10]
      kernel_size: 3
```

For diffusion models, the `EDMPrecond` wrapper is built via a `build_edm_model` factory function that also receives the backbone specification as a nested dict — see `recipes/training/resdiff.yaml` for a full example.

For graph models, `build_graph` is called separately and its output is passed to the trainer via the `graph` key in the recipe — see `recipes/training/gnn4cd_qmse.yaml`.

---

## References

<a id="ref-1"></a>[1] Baño-Medina, J., Manzanas, R., Cimadevilla, E., Fernández, J., González-Abad, J., Cofiño, A. S., & Gutiérrez, J. M. (2022). Downscaling multi-model climate projection ensembles with deep learning (DeepESD): contribution to CORDEX EUR-44. *Geoscientific Model Development Discussions*, 2022, 1–14.

<a id="ref-2"></a>[2] Fuentes–Franco, R., Krus, K., Ivanov, M., Koenigk, T., Wang, F., & Aldama‐Campino, A. (2025). Pan‐European high‐resolution downscaling using deep learning. *Journal of Geophysical Research: Machine Learning and Computation*, 2(4), e2025JH000630.

<a id="ref-3"></a>[3] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2020). Score-based generative modeling through stochastic differential equations. *arXiv preprint arXiv:2011.13456*.

<a id="ref-4"></a>[4] González-Abad, J., & Gutiérrez, J. M. (2025). Are deep learning methods suitable for downscaling global climate projections? An intercomparison for temperature and precipitation over Spain. *Artificial Intelligence for the Earth Systems*, 4(4), 240121.

<a id="ref-5"></a>[5] Blasone, V., Coppola, E., Sanguinetti, G., Arora, V., Di Gioia, S., & Bortolussi, L. (2025). Graph neural networks for hourly precipitation projections at the convection permitting scale with a novel hybrid imperfect framework. *Environmental Data Science*, 4, e47.

<a id="ref-6"></a>[6] Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the design space of diffusion-based generative models. *Advances in Neural Information Processing Systems*, 35, 26565–26577.
