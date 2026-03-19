# Deep4Production Tutorial: CORDEX-BENCH Alps Case Study with DeepESD with MLflow

This tutorial demonstrates how to use the [deep4production](https://github.com/yourorg/deep4production) framework for climate downscaling, using the CORDEX-BENCH Alps domain as a case study. We will walk through the full workflow: preparing AI-ready datasets, inspecting datasets, training a model, and running inference.

---

## 1. Introduction

**deep4production** is a command-line framework for deep learning-based climate downscaling. It operates with four main tools:

1. **d4p-datasets**: Converts NetCDF source files into AI-ready Zarr datasets containing precomputed statistics.
2. **d4p-inspect**: Inspects the created Zarr file for basic QA/QC.
3. **d4p-train**: Trains a deep learning model using configuration files.
4. **d4p-predict**: Runs inference using a pre-trained model.

All tools (except `d4p-inspect`) use YAML configuration files for reproducibility.

CORDEX-BENCH is a ...

---

## 2. Setup: Download CORDEX-BENCH Alps Data

We will use the Alps domain data from CORDEX-BENCH, available on Zenodo. Estimated size of files is: XXX GB

```python
import os
import zipfile
import shutil

os.makedirs("./data_zenodo/", exist_ok=True)

##################################
###### This line is on bash ######
!wget -P ./data_zenodo/ https://zenodo.org/records/15797226/files/ALPS_domain.zip?download=1
##################################

shutil.move("./data_zenodo/ALPS_domain.zip?download=1", "./data_zenodo/ALPS_domain.zip")

with zipfile.ZipFile('./data_zenodo/ALPS_domain.zip', 'r') as zip_ref:
        zip_ref.extractall('./data_zenodo/')
    
os.remove("./data_zenodo/ALPS_domain.zip")
```

---

## 3. Prepare AI-Ready Datasets with `d4p-create`

We will use YAML configuration files to convert the NetCDF files into Zarr format. Example configs are in `./AI_ready_datasets/configs/`.

```python
# Show example YAML config for UPSRCM (predictor)
date_init: 1961-01-01 12:00:00
date_end: 1980-12-31 12:00:00
freq: 1D

data:
  paths: [./source_files/data_zenodo/ALPS_domain/train/ESD_pseudo_reality/predictors/CNRM-CM5_1961-1980.nc]
  vars: [u_850, u_700, u_500, v_850, v_700, v_500, t_850, t_700, t_500, q_850, q_700, q_500, z_850, z_700, z_500]

output_path: ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr
overwrite: True
```

```python
# Show example YAML config for RCM (predictand)
date_init: 1961-01-01 12:00:00
date_end: 1980-12-31 12:00:00
freq: 1D

data:
  paths: [./source_files/data_zenodo/ALPS_domain/train/ESD_pseudo_reality/target/pr_tasmax_CNRM-CM5_1961-1980.nc,
          ./source_files/data_zenodo/ALPS_domain/train/ESD_pseudo_reality/predictors/Static_fields.nc]
  vars: [tasmax, pr, orog]

output_path: ./AI_ready_datasets/files/RCM_1961-1980.zarr
overwrite: True
```

Now, we produce the AI-ready datasets by running `d4p-create`:

```bash
!d4p-create ./configs/UPSRCM_1961-1980.yaml # Predictors
!d4p-create ./configs/RCM_1961-1980.yaml # Predictands
```

---

## 4. Inspect the Zarr Datasets with `d4p-inspect`

Inspect the generated Zarr files to ensure they are correct.

```bash
!d4p-inspect ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr # Predictors
!d4p-inspect ./AI_ready_datasets/files/RCM_1961-1980.zarr # Predictands
```

---

## 5. Train a Model with `d4p-train`

Prepare a YAML configuration for training. The MLflow information appears at the bottom of the YAML config file.

```python
# Show example training config
##### GENERAL INFO #####
run_ID: deepesd_mlflow
output_dir: ./outputs
overwrite: true # trains deep learning model from scratch even if a model already exists in output dir


##### TRAINING DATA CONFIGURATION (uses pre-computed zarr files) #####
data:
  load_in_memory: true # Load all data in memory for training (speeds up training if enough RAM is available)
  training_period: [1961, 1962, 1963, 1964, 1965, 1966, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1976, 1977, 1978, 1979, 1980]
  validation_period: [1967, 1975]

  predictors:
    paths: # List of paths to the predictor datasets. Can be one or more paths.
      - ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr 
    variables: [u_850, u_700, u_500, v_850, v_700, v_500, t_850, t_700, t_500, q_850, q_700, q_500, z_850, z_700, z_500]  # If null, uses all variables available in the zarrs. 
    normalizer: 
      path_reference: ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr # To standardize the input fields, the statistics from this reference dataset are used, which should be stored in the .zarr file.
      default: mean_std # Default standardization. Applies this to all the variables in the predictor set unless a specific normalizer is provided for a variable.
      q_850: std
      q_700: std
      q_500: std
    transform_to_2D: True

  predictands:
    paths: # List of paths to the predictor datasets. Can be one or more paths.
      - ./AI_ready_datasets/files/RCM_1961-1980.zarr # If null, uses all variables available in the zarrs. 
    variables:
      - pr
    normalizer: null
    transform_to_2D: True
  
  # forcings: # only accepts the following fields: "variables", "normalizer", and "operator"
  #   variables:
  #     - orog
  #   normalizer: 
  #     path_reference: ./AI_ready_datasets/files/RCM_1961-1980.zarr
  #     orog: max


##### DATA LOADER CONFIGURATION #####
dataloader:
  batch_size: 64
  shuffle: true
  num_workers: 0


##### REGRESSOR CONFIGURATION #####
model_info: 
  saving_params:
    model_save_name: DeepESD_BerGamma
    # save_every_n_epochs: 2
    # save_every_n_steps: 300
    # resume_checkpoint: DeepESD_BerGamma_epoch6.pt
  loss_params: 
    name: NLLBerGammaLoss
    module: deep4production.deep.loss
    kwargs:
      threshold: 0.999
      ignore_nans: True
  model_params:
    name: DeepESD
    module: deep4production.deep.models.cnn.DeepESD
    # kwargs model
    kwargs: # These kwargs are passed to the model's __init__ method. Check the model's code to see which kwargs it accepts.
      x_shape: [15, 16, 16] # (C, H, W). Use `d4d-datasets-inspect your_zarr_file` to get this value
      y_shape: [1, 128, 128] # (C, H, W). Use `d4d-datasets-inspect your_zarr_file` to get this value
      f_shape: [1, 128, 128]
      filters: [50, 25, 10]
      kernel_size: 3
      loss_function_name: NLLBerGammaLoss
  training_params:
    num_epochs: 1000
    patience_early_stopping: 30
    optimizer_params:
      lr: 0.0001


##### MLFlow #####
Mlflow:
  tracking_uri: https://mlflow.c3s2-384.predictia.es/
  username: banoj
  password: flow
  experiment: CSIC
  run: csic-manage-deepesd
  tags:
    domain: alps
    rcm: cnrm-aladin-6.3
    gcm: cnrm-cm5
    framework: perfect
    model: deepesd
    loss: bergamma
    contact: jorge
  save_checkpoint_every_n_epochs: null
  compute_diagnostics_every_n_epochs: 3
  diagnostics:
    scalars:
      default:
        - rmse
      pr:
        - [R01, relbiasAbs]
        - [R20, relbiasAbs]
        - [Rx1day, relbiasAbs]
        - [SDII, relbiasAbs]
        - [P98Wet, relbiasAbs]
    figures:
      on_best: true
      default:
        figure_1:
          module: deep4production.visualization.xyplots
          name: plot_psd_spatial
          kwargs:
            reshape_spatial_dims: [128, 128]
      pr:
        figure_2:
          module: deep4production.visualization.spatial
          name: plot_date_from_1D_spatial_field
          kwargs:
            date: 2095-01-01
            vmin: 0
            vmax: 10
            set_extent: [5, 15, 44, 48]
            central_longitude: 0
            cbar_label: Precipitation (mm)
            titles: [target, prediction]
            diff: True
            vminDiff: -5
            vmaxDiff: 5
            cmapDiff: BrBG
```

Train the model:

```bash
!d4p-train ./training/configs/deepesd.yaml
```

---

## 6. Run Inference with `d4p-predict`

Prepare a YAML configuration for prediction.

```python
# Show example prediction config
id_dir: ./outputs/deepesd

input_data: 
  paths: 
    - ./AI_ready_datasets/files/UPSRCM_1961-1980.zarr
  years: [1980]
  load_in_memory: true

graph: null
ensemble_size: 2

model_file: DeepESD_BerGamma_best.pt # Model at: id_dir/models/

saving_info:
  file: 1980.nc # Predictions will be saved at: id_dir/predictions/
  template: null
  formatting: null

```

Run prediction:

```bash
!d4p-predict ./inference/configs/deepesd.yaml
```

---

## 7. Summary

You have now completed the full deep4production workflow for the CORDEX-BENCH Alps domain:
- Downloaded and prepared data
- Inspected AI-ready datasets
- Trained a deep learning model
- Performed inference

For more details, see the [deep4production README](../deep4production/README.md) and the [CORDEX-BENCH documentation](https://github.com/CORDEX-BENCH).

---

## 8. References

- [deep4production GitHub](https://github.com/yourorg/deep4production)
- [CORDEX-BENCH Zenodo](https://zenodo.org/record/XXXXXX)