## Load libraries
import os
import sys
import yaml
import zarr
import string
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import torch
import importlib

## Deep4downscaling
import deep4downscaling as d4d
from deep4downscaling.trans import compute_valid_mask
from deep4downscaling.console.d4dpredict import d4dpredict
from deep4downscaling.deep.pred import _pred_to_xarray


##################################################################################################################################
##################################################################################################################################

def read_metadata_from_yaml(yaml_path: str) -> dict:
    """
    Reads a YAML file and returns its content as a dictionary.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML file.

    Returns
    -------
    dict
        Contents of the YAML file.
    """
    with open(f"{yaml_path}", "r") as f:
        metadata = yaml.safe_load(f)
    return metadata




def forward_regressor_get_cond(x_data, model, H, W):
    model.eval()
    with torch.no_grad():
        regressor_outputs = model(x_data)
        num_gp = regressor_outputs.shape[1] // 3

    # Get parameters
    p = regressor_outputs[:,:num_gp].reshape(H, W)
    shape = torch.exp(regressor_outputs[:,num_gp:(2*num_gp)]).reshape(H, W)
    scale = torch.exp(regressor_outputs[:,(2*num_gp):]).reshape(H, W)
    mean = shape * scale  # mean of BerGamma
    var = shape * (scale ** 2)  # variance of BerGamma
    
    return p.unsqueeze(0), mean.unsqueeze(0), var.unsqueeze(0)  # add batch dimension and return




##############################################################################################################################
##############################################################################################################################
def d4dpredictREGGEN(
    input_data: dict,
    regressor: dict,
    generator: dict,
    template_path: str = None,
    ensemble_size: int = 1,
    output: str = None
):

    # --- Parsing metadata ----------------------------------
    ####################################################################################
    metadata_yaml = generator["metadata_yaml"]
    metadata = read_metadata_from_yaml(metadata_yaml)
    metadata_regressor_yaml = regressor["metadata_yaml"]
    metadata_regressor = read_metadata_from_yaml(metadata_regressor_yaml)


    # --- Load regressor model ----------------------------------
    ####################################################################################
    model_architecture = metadata_regressor["architecture"]
    print(f"Loading regressor model architecture: {model_architecture}")
    module = importlib.import_module("deep4downscaling.deep.models") # Dynamically import from module
    model_func = getattr(module, model_architecture)
    regressor_model = model_func(**metadata_regressor["model_parameters"])
    model_path = regressor["model_path"]
    regressor_model.load_state_dict(torch.load(model_path))

    # --- Load denoiser model ----------------------------------
    ####################################################################################
    model_architecture = metadata["architecture"]
    print(f"Loading generator model architecture: {model_architecture}")    
    module = importlib.import_module("deep4downscaling.deep.models") # Dynamically import from module
    model_func = getattr(module, model_architecture)
    denoiser_model = model_func(**metadata["model_parameters"])
    model_path = generator["model_path"]
    denoiser_model.load_state_dict(torch.load(model_path))


    # --- Metadata xarray object ----------------------------------
    ####################################################################################
    var_target = metadata["var_target"]
    template = xr.open_dataset(template_path)[var_target]
    mask = compute_valid_mask(template)
    
    spatial_dims = ["lat", "lon"]
    if "x" in mask.dims and "y" in mask.dims:
      spatial_dims = ["y", "x"]


    # --- Hardware ----------------------------------
    ####################################################################################
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Display info ----------------------------------
    ####################################################################################
    print(f"""
    -----------------------------------------------------------------------------------------------------------
    WELCOME TO D4D PREDICTION MODULE! 📈🤖📊

    REGRESSOR:
        Model: {regressor["model_path"]}
        Metadata: {regressor["metadata_yaml"]}

    GENERATOR:
        Model: {generator["model_path"]}
        Metadata: {generator["metadata_yaml"]}
        Number of steps: {generator["kwargs"].get("N_steps", None)}
        Guidance scale: {generator["kwargs"].get("guidance_scale", None)}

    GENERAL INFO:
        Prediction(s) will be saved here: {output}
        Device: {device}
        Ensemble size: {ensemble_size}
    ---------
    """)  

    # --- Noise schedule ----------------------------------
    ####################################################################################
    N_steps = generator["kwargs"].get("N_steps", 50)
    sigma_min = metadata["sigma_min"]
    sigma_max = metadata["sigma_max"]
    sigmas = torch.exp(torch.linspace(np.log(sigma_max), np.log(sigma_min), N_steps)).to(device) # log sigma_t = N(P_mean, P_sigma ** 2)

    # --- Load input data to the REGRESSOR ----------------------------------
    ####################################################################################
    ds = zarr.open(input_data["path"], mode='r')
    num_samples = ds.shape[0]
    dates_test = [ date for date in ds.attrs.get("dates") if int(date[:4]) in input_data["years"] ]
    samples_idx = [i for i, date in enumerate(ds.attrs.get("dates")) if int(date[:4]) in input_data["years"] ]
    if input_data["variables"] is None:
        vars = ds.attrs.get("variables")

    # --- Iterate over test samples ----------------------------------
    ####################################################################################
    regressor_model.to(device)
    denoiser_model.to(device)
    print("🚀 Starting REGRESSOR + GENERATOR prediction... 🚀")
    pred = []
    for i, sample_idx in enumerate(samples_idx):
        date = ds.attrs.get("dates")[sample_idx]
        print(f"Inference for sample: {date} ---- ({i+1}/{len(dates_test)})")

        # --- Start from max noise ----------------------------------
        ####################################################################################
        with torch.no_grad():
            x = torch.randn(mask["pr"].values.shape) * sigma_max  
            x = x.unsqueeze(0).unsqueeze(0) # add batch and channel dimension
            x = x.to(device)

        # --- Context data (large-scale predictors) ----------------------------------
        ####################################################################################
        num_vars = ds.shape[1]
        m_low = np.array(metadata["mean_input"]).reshape(num_vars, 1, 1)
        s_low = np.array(metadata["std_input"]).reshape(num_vars, 1, 1)
        x_cond_low = ds[sample_idx].astype(np.float32)
        x_cond_low = (x_cond_low - m_low) / s_low
        x_cond_low = torch.tensor(x_cond_low, dtype=torch.float32, device=device).unsqueeze(0)  # shape [C, H, W] or [batch, C, H, W]
        # print("x_cond_low shape:", x_cond_low.shape)
        
        # --- Context data (regressor predictions) ----------------------------------
        ####################################################################################
        p, mean, var = forward_regressor_get_cond(
            x_data = x_cond_low, # add batch dimension
            model = regressor_model,
            H = mask["pr"].values.shape[0],
            W = mask["pr"].values.shape[1]
        )     

        num_vars_high = 2
        m_high = torch.tensor(np.array(metadata["mean_residual"])[2:4].reshape(num_vars_high, 1, 1)).to(device) # metadata["mean_residual"] contains shape, scale, mu, std, residual. Selecting idx = 2 and 3.
        s_high = torch.tensor(np.array(metadata["std_residual"])[2:4].reshape(num_vars_high, 1, 1)).to(device) # metadata["mstd_residual"] contains shape, scale, mu, std, residual. Selecting idx = 2 and 3.

        x_cond_high = torch.cat([mean, var], dim=1).squeeze(0) # shape [B, C, H, W]
        x_cond_high = (x_cond_high - m_high) / s_high # standardize
        x_cond_high = x_cond_high.to(device=device, dtype=torch.float32).unsqueeze(0) # shape [B, C, H, W] 
        # print("x_cond_high shape:", x_cond_high.shape)

        # --- Reverse SDE sampling ----------------------------------
        ####################################################################################
        n_steps = len(sigmas) - 1 
        for i in range(n_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # --- Compute preconditioning factors ----------------------------------
            ####################################################################################
            sigma_data = metadata["sigma_data"]
            c_in = 1.0 / torch.sqrt(sigma_data ** 2 + sigma ** 2)
            c_skip = (sigma_data ** 2) / (sigma ** 2 + sigma_data ** 2)
            c_out  = (sigma * sigma_data) / torch.sqrt(sigma_data ** 2 + sigma ** 2)
            c_noise = 0.25 * torch.log(sigma) * torch.ones((x.shape[0], 1, 1, 1), device=device)
            # print("c_noise shape:", c_noise.shape)
            
            # --- Compute scores ----------------------------------
            ####################################################################################
            denoised_cond = denoiser_model(c_in * x, x_cond_low, x_cond_high, c_noise) # conditional score
            denoised_uncond = denoiser_model(c_in * x, x_cond_low, x_cond_high, c_noise, force_uncond=True) # Unconditional score

            # --- Classifier-free guidance ----------------------------------
            ####################################################################################
            guidance_scale = generator["kwargs"].get("guidance_scale", 0.0)
            denoised = denoised_uncond + guidance_scale * (denoised_cond - denoised_uncond)
            denoised = c_skip * x + c_out * denoised

            # --- Compute EDM update (Euler-Maruyama) ------------------------------------------------
            ####################################################################################
            d = (x - denoised) / sigma                    # ODE drift term
            dt = sigma_next - sigma                       # step in σ (negative)
            noise = torch.randn_like(x)                   # standard Gaussian noise
            x = x + d * dt # + torch.sqrt(torch.clamp(sigma_next**2 - sigma**2, min=0)) * noise # Euler-Maruyama update
            
        # --- Rescale Denoiser output ----------------------------------
        ####################################################################################
        x = x.cpu().detach().numpy()
        num_vars_residual = 1
        m_residual = np.array(metadata["mean_residual"])[4].reshape(num_vars_residual, 1, 1) # metadata["mean_residual"] contains shape, scale, mu, std, residual. Only selecting residual, idx = 4.
        s_residual = np.array(metadata["std_residual"])[4].reshape(num_vars_residual, 1, 1) # metadata["std_residual"] contains shape, scale, mu, std, residual. Only selecting residual, idx = 4.
        x = x * s_residual + m_residual  # destandardize

        # --- Add regressor and generator output in a single object ----------------------------------
        ####################################################################################
        x = x + mean.cpu().numpy()  # add regressor mean prediction
        x = x + regressor["kwargs"].get("threshold", 0.0)  # add threshold back (precipitation NLL transformation)
        x = np.clip(x, a_min=0, a_max=None)  # avoid negative precipitation

        # --- Multiply by deterministic estimate of the occurrence of precipitation ----------------------------------
        ####################################################################################
        b = "assfasf"
        # print("x shape:", x.shape)
        # print("p shape:", p.shape)
        # x = x * b # b is binary mask of precipitation occurrence from regressor

        # --- Convert to xarray using "mask" as template ----------------------------------
        ####################################################################################
        ds_sample = mask.copy()
        ds_sample[var_target[0]].values = x.squeeze()
        ds_sample = ds_sample.expand_dims(time=[np.datetime64(date)])
        # ds_sample = ds_sample.ffill(dim='time')
        # print("Sample prediction ---")
        # print(ds_sample)
        pred.append(ds_sample)

    ## --- Concatenate samples along dimension "time" ----------------------------------
    ####################################################################################
    pred = xr.concat(pred, dim = "time")
    print(pred)
    
    ## --- Save prediction ----------------------------------
    ####################################################################################
    os.makedirs(os.path.dirname(output), exist_ok=True)
    pred.to_netcdf(output) # Save prediction
    print("✅  🤞 🎯 Prediction REGRESSOR + GENERATOR finished successfully! 🎯  🤞 ✅")
    print(f"✅  🤞 🎯 Prediction saved at: {output}  🎯  🤞 ✅")


    


    