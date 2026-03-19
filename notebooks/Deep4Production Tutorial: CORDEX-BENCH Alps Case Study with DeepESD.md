# Deep4Production Tutorial: CORDEX-BENCH Alps Case Study with DeepESD

This tutorial demonstrates how to use the [deep4production](https://github.com/yourorg/deep4production) framework for climate downscaling, using the CORDEX-BENCH Alps domain as a case study. We will walk through the full workflow: preparing AI-ready datasets, inspecting datasets, training a model, and running inference.

---

## 1. Introduction

**deep4production** is a command-line framework for deep learning-based climate downscaling. It operates with four main tools:

1. **d4p-datasets**: Converts NetCDF source files into AI-ready Zarr datasets containing precomputed statistics.
2. **d4p-inspect**: Inspects the created Zarr file for basic QA/QC.
3. **d4p-train**: Trains a deep learning model using configuration files.
4. **d4p-predict**: Runs inference using a pre-trained model.

All tools (except `d4p-inspect`) use YAML configuration files for reproducibility.


---

## 2. Case study: CORDEX-BENCH

**CORDEX-BENCH** is a community benchmark dataset designed to evaluate machine learning methods for climate downscaling in a standardized and reproducible way. 

It builds on the broader **CORDEX (Coordinated Regional Climate Downscaling Experiment)** initiative, but introduces key features tailored for ML:

**Key ideas behind CORDEX-BENCH:**

* **Pseudo-reality setup**: Instead of using observations directly, CORDEX-BENCH treats data from **Regional Climate Models** as the "groundtruth" and a corresponding coarse-resolution version.

* **Predictors vs Predictands**

  * **Predictors (X):** Large-scale atmospheric variables (e.g., winds, temperature, geopotential height)

  * **Predictands (Y):** Local surface variables (e.g., precipitation, temperature)

* **Standardized domains**: Includes three geographic regions (Central Europe, New Zealand, and South Africa), enabling fair comparison across methods.

* **Consistent splits**: Training, validation, and testing periods are predefined.

The reader is referred to CORDEX-BENCH official site for more details. https://github.com/WCRP-CORDEX/ml-benchmark
Here for the purpose of simplicity since the objective is to illustrate the functionining of `deep4production`, we plan to conduct a simplified experiment from CORDEX-BENCH with the following characteristics:

* **Domain:** Central Europe
* **Predictors:**
  * **Dataset**: Upscaled CNRM-CM5-ALADIN-63 Regional Climate Model
  * **Spatial resolution (dimensions)**: 2-degrees (16 x 16)
  * **Temporal resolution**: daily
  * **Variables:** 15
    * `z_850`: geopotential at 850 hPa
    * `z_700`: geopotential at 700 hPa
    * `z_500`: geopotential at 500 hPa
    * `t_850`: air temperature at 850 hPa
    * `t_700`: air temperature at 700 hPa
    * `t_500`: air temperature at 500 hPa
    * `q_850`: specific humidity at 850 hPa
    * `q_700`: specific humidity at 700 hPa
    * `q_500`: specific humidity at 500 hPa
    * `u_850`: zonal wind at 850 hPa
    * `u_700`: zonal wind at 700 hPa
    * `u_500`: zonal wind at 500 hPa
    * `v_850`: meridional wind at 850 hPa
    * `v_700`: meridional wind at 700 hPa
    * `v_500`: meridional wind at 500 hPa
* **Predictands:** 
  * **Dataset**: CNRM-CM5-ALADIN-63 Regional Climate Model
  * **Spatial resolution (dimensions)**: 0.11-degrees (128 x 128)
  * **Temporal resolution**: daily
  * **Variables:** 1
    * `pr`: precipitation
* **Forcings:** 
  * **Dataset**: CNRM-CM5-ALADIN-63 Regional Climate Model
  * **Spatial resolution (dimensions)**: 0.11-degrees (128 x 128)
  * **Temporal resolution**: daily
  * **Variables:** 1
    * `orog`: orography

For the sake of simplicity hereafet CNRM-CM5-ALADIN-63 Regional Climate Model and Upscaled CNRM-CM5-ALADIN-63 Regional Climate Model will be reffered to as RCM and UPSRCM, respectively

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

Prepare a YAML configuration for training.

```python
# Show example training config
##### GENERAL INFO #####
run_ID: deepesd
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
