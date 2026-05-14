# Loss Functions

This package contains all loss functions available in **deep4production**, organized by family. Every class is re-exported from the top-level `__init__.py`, so YAML recipes can always reference them as `module: deep4production.deep.loss` regardless of which file they live in.

______________________________________________________________________

## Available loss functions

### Standard regression (`standard.py`)

Point-estimate losses for deterministic downscaling.

| Class | Description |
|---|---|
| `MaeLoss` | Mean Absolute Error. Supports NaN masking. |
| `MseLoss` | Mean Squared Error. Supports NaN masking. |
| `QuantisedMSELoss` | MSE + quantile-weighted MSE (QMSE). Penalises errors in extreme quantile bins more heavily. Requires a reference Zarr path to compute bin edges. |

______________________________________________________________________

### Negative log-likelihood (`nll.py`)

Probabilistic losses for models that output distribution parameters rather than point estimates.

| Class | Description |
|---|---|
| `NLLGaussianLoss` | NLL for a Gaussian distribution. Expects the model to output `(mean, log_var)` per grid point. |
| `NLLBerGammaLoss` | NLL for a Bernoulli-Gamma distribution. Suitable for precipitation: models the occurrence probability jointly with the intensity distribution. Expects `(p, log_shape, log_scale)` per grid point. |

______________________________________________________________________

### Asymmetric precipitation loss (`asym.py`)

Specialized loss for precipitation downscaling that applies higher penalties on underestimation of heavy events.

| Class | Description |
|---|---|
| `Asym` | Combines MAE with a CDF-weighted asymmetric penalty term. Gamma distribution parameters are fitted once from a reference dataset (Zarr or NetCDF) and cached on disk. Supports `per_year` or `full`-period fitting. |

______________________________________________________________________

### Continuous Ranked Probability Score (`crps.py`)

Ensemble scoring rules evaluated in both the spatial and spectral domains.

| Class | Description |
|---|---|
| `CRPSSpectralLoss` | Fair CRPS over the spatial field plus a spectral CRPS term computed via 2-D FFT. The spectral term can be optionally low-pass filtered at a given spatial resolution. Accepts a list of ensemble members as `output`. |

______________________________________________________________________

### Diffusion model losses (`diffusion.py`)

Losses designed specifically for score-based / EDM diffusion trainers.

| Class | Description |
|---|---|
| `WeightedDenoisingScoreMatchingLoss` | EDM-weighted denoising score-matching loss (Karras et al. 2022). Expects the model output to already be the denoised prediction `D_θ` from an `EDMPrecond` wrapper. The per-sample noise level `sigma_t` is passed as a third argument to `forward`. |

______________________________________________________________________

### Binary classification losses (`classification.py`)

Losses for models that predict precipitation occurrence or other binary fields.

| Class | Description |
|---|---|
| `BinaryCrossEntropyLoss` | Binary cross-entropy with logits. Binarizes the target at a configurable threshold (single value or per-channel list). Supports NaN masking. |
| `BernoulliFocalLoss` | Focal loss variant of BCE (Lin et al. 2017). Down-weights easy negatives via a focusing parameter `gamma`. Suitable for datasets with heavy class imbalance between wet and dry days. |

______________________________________________________________________

## Recipe usage

Reference any loss by name under `model_info.loss_params` in a training YAML:

```yaml
model_info:
  loss_params:
    name: Asym                          # class name
    module: deep4production.deep.loss   # always the top-level package
    kwargs:
      ref_path: /path/to/y.zarr
      var: pr
      ignore_nans: true
      asym_path: /path/to/output/asym_params
```

For losses with a non-standard `forward` signature (e.g. `WeightedDenoisingScoreMatchingLoss` takes `sigma_t`), the trainer is responsible for supplying the extra arguments — no change is needed in the recipe.
