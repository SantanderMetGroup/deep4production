# Deep4Production Tutorial: CORDEX-BENCH Alps Case Study with DeepESD (Explainability)

This tutorial extends the [DeepESD CORDEX-BENCH Alps tutorial](../Deep4Production%20Tutorial%3A%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD/Deep4Production%20Tutorial%3A%20CORDEX-BENCH%20Alps%20Case%20Study%20with%20DeepESD.md) with a fifth CLI tool, **`d4p-explain`**, which computes **gradient-based input attribution (saliency)** for a trained downscaler.

The question it answers: *which predictor channels (and where on the predictor grid) does the model rely on to produce a given predictand?* This is a practical diagnostic — e.g. to compare a univariate vs a multivariate emulator, or **perfect** (UPSRCM) vs **imperfect** (GCM) predictors, and to test hypotheses such as *"the model over-relies on the humidity channels, which carry the largest GCM bias."*

______________________________________________________________________

## 0. What `d4p-explain` computes

For a chosen predictand `target_var`, the explainer builds a scalar `S` by reducing that variable over a **predictand box or gridpoint** you specify, and differentiates `S` with respect to the predictor input:

- `saliency` = `dS / d(predictor)` — one signed map per date, per predictor channel.
- `channel_importance` — per-date mean `|saliency|` per channel.
- `saliency_percent` / `channel_contribution_percent` — a **magnitude-robust** view where each date's `|saliency|` is renormalised to sum to 100% across all predictor pixels and variables (removes day-to-day magnitude swings).

Two knobs control the spaces the gradient lives in:

- `input_space`: `normalized` (standardised inputs → channels are unit-comparable) or `raw` (physical predictor units).
- `target_space`: `physical` (denormalised predictand, e.g. mm — **comparable across models**) or `model` (raw network output).

> **Validity.** Attribution differentiates the model's *direct output channels*, which equal the predictands only for **deterministic** regressors (MSE / asymmetric loss). Distributional-loss checkpoints (BerGamma / Gaussian) emit *distribution parameters*, so the explainer **refuses** them. This tutorial therefore uses a **DeepESD trained with MSE loss**.

`d4p-explain` is driven by a **YAML configuration file**, like every other d4p tool, and writes its output to a per-model `xai/` directory:

```
outputs/
└── deepesd_mse/
    ├── models/        # trained models (.pt)
    ├── predictions/   # d4p-downscale output (.nc)
    └── xai/           # d4p-explain output (.nc)   <-- new
```

______________________________________________________________________

## 1. Prerequisites

We assume you have already followed the base tutorial up to training, but with the **MSE** recipe (`recipes/training/deepesd_mse.yaml`) so the predictand `pr` is regressed directly. Concretely, this notebook expects:

- AI-ready predictors at `./AI_ready_datasets/files/UPSRCM_1961-1980.zarr` (and, optionally, a GCM-driven file for the imperfect case).
- A trained checkpoint at `./outputs/deepesd_mse/models/DeepESD_best.pt`.
- (Optional) a predictand template `./templates/pr_template.nc` to mask NaN gridpoints.

The 15 predictor channels (CORDEX-BENCH Alps) are, in order:

```
u_850 u_700 u_500   v_850 v_700 v_500   t_850 t_700 t_500   q_850 q_700 q_500   z_850 z_700 z_500
```

`q_*` are specific humidity — the channels we most expect a GCM-driven (imperfect) run to be sensitive to.

```python
import os, yaml
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Run everything relative to the tutorial's example/ directory.
# os.chdir('/path/to/example')   # <-- set if needed
ID_DIR    = './outputs/deepesd_mse'
X_PERFECT = './AI_ready_datasets/files/UPSRCM_1961-1980.zarr'
# X_IMPERFECT = './AI_ready_datasets/files/GCM_1961-1980.zarr'   # for the perfect-vs-imperfect comparison
TEMPLATE  = './templates/pr_template.nc'   # optional; set to None to skip masking
os.makedirs('./explain/configs', exist_ok=True)
```

______________________________________________________________________

## 2. Write the `d4p-explain` configuration

As with every d4p tool, the run is fully described by a YAML file. We target `pr`, reduce over a central index box of the 128×128 predictand grid, and ask for the percentage view.

```python
explain_cfg = {
    'id_dir': ID_DIR,
    'input_data': {'paths': [X_PERFECT], 'years': [1980], 'load_in_memory': True},
    'graph': None,
    'ensemble_size': 1,
    'model_file': 'DeepESD_best.pt',          # at id_dir/models/
    'saving_info': {'file': 'xai_pr_1980_perfect.nc', 'template': TEMPLATE},  # saved at id_dir/xai/
    # DeepESD forward is model(x, f) -> base Explainer (no architecture subclass needed).
    'd4p_explainer': {
        'name': 'Explainer',
        'module': 'deep4production.core.explainers.explainer',
    },
    'explain_params': {
        'method': 'gradient',
        'target_var': 'pr',
        'reduction': 'mean',
        'input_space': 'normalized',          # 'raw' for physical predictor units
        'target_space': 'physical',           # denormalised pr (comparable across models)
        'percent': True,
        'batch_size': 8,
        # Predictand box the gradient is computed over. Alternatives:
        #   whole domain:    {'type': 'all'}
        #   single point:    {'type': 'point', 'i': 64, 'j': 64}
        #   geographic box:  {'type': 'box', 'lat': [44, 48], 'lon': [5, 15]}  (needs 1D lats_y/lons_y)
        'target_region': {'type': 'box', 'i': [40, 88], 'j': [40, 88]},
    },
}
cfg_path = './explain/configs/deepesd_mse_explain.yaml'
with open(cfg_path, 'w') as f:
    yaml.safe_dump(explain_cfg, f, sort_keys=False)
print('wrote', cfg_path)
```

The same recipe lives, in template form, at `recipes/explain/deepesd_mse.yaml`.

______________________________________________________________________

## 3. Run the attribution

Either from the **CLI**…

```bash
d4p-explain ./explain/configs/deepesd_mse_explain.yaml
```

…or directly through the **Python API** (handy for sweeping regions / dates in-notebook). This is exactly what the CLI does under the hood:

```python
from deep4production.core.explainers.explainer import Explainer

ex = Explainer(
    id_dir=explain_cfg['id_dir'],
    input_data=explain_cfg['input_data'],
    model_file=explain_cfg['model_file'],
    saving_info=explain_cfg['saving_info'],
    graph=explain_cfg['graph'],
    ensemble_size=explain_cfg['ensemble_size'],
)
ds = ex.explain(**explain_cfg['explain_params'])   # also writes id_dir/xai/<file>
ds
```

______________________________________________________________________

## 4. Load and inspect the attribution dataset

```python
xai_path = os.path.join(ID_DIR, 'xai', explain_cfg['saving_info']['file'])
ds = xr.open_dataset(xai_path)
print(ds)
print('\nattrs:', dict(ds.attrs))
```

The dataset carries `saliency (time, var_x, y_x, x_x)`, the per-channel `channel_importance (time, var_x)`, and the percentage view `saliency_percent` / `channel_contribution_percent`.

______________________________________________________________________

## 5. Which predictor channels does DeepESD rely on?

`channel_contribution_percent` is the headline: the per-date share (%) of total attribution carried by each predictor channel. Averaging over the 1980 test dates gives a single, magnitude-robust ranking.

```python
chan = ds['channel_contribution_percent'].mean('time')   # (var_x,)
order = chan.to_series().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 4))
colors = ['#D9532C' if v.startswith('q_') else '#1F5C99' for v in order.index]
ax.bar(order.index, order.values, color=colors)
ax.set_ylabel('mean contribution [%]')
ax.set_title('DeepESD — predictor-channel contribution to pr (perfect predictors, 1980)')
ax.tick_params(axis='x', rotation=60)
plt.tight_layout(); plt.show()

print('Humidity (q_*) share: %.1f%%' % order[[v for v in order.index if v.startswith('q_')]].sum())
```

______________________________________________________________________

## 6. Where on the predictor grid? A saliency map

`saliency_percent` keeps the spatial structure: for a chosen channel we can see *which large-scale gridpoints* drive the prediction. Here we average `|saliency_percent|` over the test dates for the leading humidity channel.

```python
channel = order.index[0]   # most important channel; or pick e.g. 'q_850'
m = np.abs(ds['saliency_percent'].sel(var_x=channel)).mean('time')   # (y_x, x_x)

fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(m.values, origin='lower', cmap='magma')
ax.set_title(f'Mean |saliency_percent| — {channel}')
ax.set_xlabel('predictor x-index'); ax.set_ylabel('predictor y-index')
fig.colorbar(im, ax=ax, label='% of total attribution')
plt.tight_layout(); plt.show()
```

______________________________________________________________________

## 7. Perfect vs imperfect — the diagnostic that matters

To test the *humidity-reliance under GCM bias* hypothesis, rerun the steps above pointing `input_data.paths` at a **GCM-driven** zarr (the imperfect case) and a different `saving_info.file` (e.g. `xai_pr_1980_imperfect.nc`), then compare `channel_contribution_percent`:

```python
# explain_cfg['input_data']['paths'] = [X_IMPERFECT]
# explain_cfg['saving_info']['file'] = 'xai_pr_1980_imperfect.nc'
# ... rewrite YAML, re-run d4p-explain, reload, and overlay the two bar charts.
```

If the humidity (`q_*`) share **rises** going from perfect to imperfect — especially for a multivariate model relative to a univariate one — that is direct evidence the model leans on the GCM-biased moisture channels, motivating fixes such as per-model bias correction, humidity-channel robustness augmentation, or a domain-invariant encoder.

> Tip: keep `target_space: physical` for any cross-model comparison, so models with different predictand normalizers (e.g. univariate raw units vs multivariate min-max) are on the same scale.

______________________________________________________________________

## 8. Summary

- `d4p-explain` adds gradient input-attribution to the d4p CLI, driven by a YAML recipe like every other tool.
- It reports per-channel **contribution %** and per-pixel **saliency maps**, in physical or normalized spaces, reduced over a **predictand box or gridpoint** you choose.
- It is valid for **deterministic** regressors (MSE/asym DeepESD, SongUNet); distributional-loss checkpoints are refused.
- The intended use here is **perfect-vs-imperfect** and **uni-vs-multivariate** comparisons of predictor reliance.

Recipes: `recipes/explain/deepesd_mse.yaml`, `recipes/explain/song_unet_det.yaml`.

______________________________________________________________________

## 9. References

- [CORDEX-BENCH Github](https://github.com/WCRP-CORDEX/ml-benchmark)
- [CORDEX-BENCH Zenodo](https://zenodo.org/records/17957264)
- [DeepESD](https://gmd.copernicus.org/articles/15/6747/2022/)
