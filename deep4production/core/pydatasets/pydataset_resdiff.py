import os
import importlib
import numpy as np
import xarray as xr 
import torch
import zarr
import yaml
from torch import from_numpy
## Deep4production
from deep4production.core.datasets.dataset import dataset
from deep4production.core.pydatasets.pydataset import pydataset
from deep4production.utils.trans import from_pred_to_xarray
from deep4production.deep.utils import load_model
########################################################################################################
class pydataset_custom(pydataset):
    """
    Custom dataset class for residual-based deep learning models.
    Purpose: Loads predictors, predictands, regressor model, computes and stores residuals, and provides context for model training.
    Parameters:
        predictors (dict): Predictor dataset configuration.
        predictands (dict): Predictand dataset configuration.
        temporal_period (list): List of target dates.
        dataset (str): 'training' or 'validation'.
        path_regressor (str): Path to regressor model.
        residuals (dict): Residuals configuration.
        load_in_memory (bool): Whether to load all data into memory.
        add_pred_mean (bool): Whether to add deterministic prediction as context.
        add_context_lowres (bool): Whether to add low-res predictors as context.
    """
    def __init__(self, 
                predictors: dict,
                predictands: dict,
                temporal_period: list,
                dataset: str = "training",
                path_regressor: str = None,
                residuals: dict = None,
                load_in_memory: bool = True,
                add_pred_mean: bool = True,
                add_context_lowres: bool = True):          

        # --- Call parent constructor to initialize common attributes ----------------------------------------------------------------------
        super().__init__(
            predictors=predictors,
            predictands=predictands,
            temporal_period=temporal_period,
            load_in_memory=load_in_memory
        )

        # --- Add generator-specific initialization ----------------------------------------------------------------------
        # -- Add context to the model? Options: deterministic prediction (high-res) and large-scale predictor variables (low-res) --
        self.add_pred_mean = add_pred_mean
        self.add_context_lowres = add_context_lowres

        # -- Regressor --
        print("📦 Loading regressor model...")
        self.regressor_model = load_model(path = path_regressor)

        # -- Residuals --
        self.template = xr.open_dataset(residuals["template"])
        variables_residuals = [f"{target}_residual" for target in self.vars_predictands] + [f"{target}_normalized" for target in self.vars_predictands]
        path_residuals_zarr = f"{residuals["path"][:-5]}_{dataset}.zarr"    
        if not os.path.exists(path_residuals_zarr):
            print(f"💾 Producing residuals (netcdf and zarr files)")
            for idx, date in enumerate(self.dates_common):
                print(date)
                # Netcdf
                self.forward_pass_regressor(f"./aux_residuals_{idx}.nc", date = date)
            # Zarr
            dataset(date_init = self.dates_common[0], 
                        date_end = self.dates_common[-1], 
                        freq = self.freq, 
                        data = {"paths": [f"./aux_residuals_{idxn}.nc" for idxn, date in enumerate(self.dates_common)], "vars": variables_residuals}).to_disk(path_residuals_zarr)
            [os.remove(f"./aux_residuals_{idxn}.nc") for idxn, date in enumerate(self.dates_common)]
        else: 
            print(f"✅ Residuals (zarr file) is already available at: {path_residuals_zarr}. Skipping computation.")
        # Update idx in sample_map which contains the indexing information for the common dates.
        self.sample_map = self.update_sample_map()
        
        # -- Delete objects -- 
        self.template.close()
        del self.template

        # -- Load in memory? --
        self.r = [zarr.open(path_residuals_zarr, mode='r')] # Open zarr file of residuals
        if self.load_in_memory:
            print("📦 Loading residuals into memory for faster access...")
            self.r_data = [np.array(r) for r in self.r]

    # -------------------------------------------------------------------------
    def update_sample_map(self):
        """
        Updates sample map with additional indexing for residuals.
        Returns:
            dict: Updated sample map.
        """
        sm = self.sample_map
        for i in range(self.num_samples):
            sm[i].extend([0, i])
        return sm

    # -------------------------------------------------------------------------
    def forward_pass_regressor(self, path, date):
        """
        Runs regressor model for a given date and saves output and residuals to NetCDF.
        Parameters:
            path (str): Output NetCDF path.
            date: Target date.
        Returns:
            None
        """
        # -- Get sample --
        i_X, j_X = self.get_idx_sample(date, self.x)
        i_Y, j_Y = self.get_idx_sample(date, self.y)
        source_x = self.x[i_X][j_X]
        source_y = self.y[i_Y][j_Y].astype(np.float32)
        x = from_numpy(source_x)[self.idx_vars_x]
        # print(f"x output: {x.shape}")
        y = from_numpy(source_y)[self.idx_vars_y]
        # print(f"y output: {y.shape}")

        # --- Normalize (X) ---
        if self.normalizer_x is not None:
            for i, variable in enumerate(self.vars_predictors):
                normalizer_func = get_func_from_string(self.normalizer_x["module"], self.normalizer_x["normalizer_func_per_variable"][variable])
                x[i,:] = normalizer_func(x[i,:], **self.normalizer_x["kwargs"][variable])

        # --- Normalize (Y) ---
        if self.normalizer_y is not None:
            for i, variable in enumerate(self.vars_predictands):
                normalizer_func = get_func_from_string(self.normalizer_y["module"], self.normalizer_y["normalizer_func_per_variable"][variable])
                y[i,:] = normalizer_func(y[i,:], **self.normalizer_y["kwargs"][variable])

        # --- Transform to 2D (X) ---
        if self.transform_to_2D_x:
            C, G = x.shape
            x = x.reshape(C, self.H_x, self.W_x) # Shape (C, H, W)
        x = x.unsqueeze(0) # Use unsqueeze to add singleton dimension along the samples.
        # print(f"x: {x.shape}")

        # --- Transform to 2D (Y) ---
        if self.transform_to_2D_y:
            C, G = y.shape
            y = y.reshape(C, self.H_y, self.W_y) # Shape (C, H, W)
        y = y.unsqueeze(0) # Use unsqueeze to add singleton dimension along the samples.
        # print(f"y: {y.shape}")

        # -- Deterministic prediction --
        with torch.no_grad():
            regressor_output = self.regressor_model(x) 
            # print(f"Regressor output: {regressor_output.shape}")
            
        # -- Residual --
        residual = np.array(y) - np.array(regressor_output)
        
        # -- Residual and deterministic prediction to xarray --
        residual = xr.merge([from_pred_to_xarray(residual[0,i,...][None, ...], date, f"{var_target}_residual", self.template) 
                                for i, var_target in enumerate(self.vars_predictands)])
        regressor_output = xr.merge([from_pred_to_xarray(regressor_output[0,i,...][None, ...], date, f"{var_target}_normalized", self.template) 
                                for i, var_target in enumerate(self.vars_predictands)])

        # -- Merge and save --
        out = xr.merge([residual, regressor_output])
        out.to_netcdf(path)
        return None

    # -------------------------------------------------------------------------
    def __getitem__(self, idx):
        """
        Returns a tuple (residual, c_low, c_high) for a given sample index.
        Parameters:
            idx (int): Sample index.
        Returns:
            tuple: (residual, c_low, c_high)
        """
        # -- Get sample --
        idx_zarr_X, idx_real_X, idx_zarr_Y, idx_real_Y, idx_zarr_R, idx_real_R = self.sample_map[idx]
        # print(f"{idx_zarr_X} - {idx_real_X} - {idx_zarr_Y} - {idx_real_Y} - {idx_zarr_R} - {idx_real_R}")
        if self.load_in_memory:
            X = self.x_data[idx_zarr_X]
            R = self.r_data[idx_zarr_R]
        else:
            X = self.x[idx_zarr_X]
            R = self.r[idx_zarr_R]
        source_x = from_numpy(X[idx_real_X].astype(np.float32))[self.idx_vars_x]
        # print(f"source_x: {source_x.shape}")
        source_r = from_numpy(R[idx_real_R].astype(np.float32))
        # print(f"source_r: {source_r.shape}")

        # -- Residual --
        num_vars = len(self.vars_predictands)
        residual = source_r[:num_vars,:]
        # --- Transform to 2D (residual) ---
        if self.transform_to_2D_y:
            C, G = residual.shape
            residual = residual.reshape(C, self.H_y, self.W_y) # Shape (C, H, W)
        # print(f"Residual: {residual.shape}")
        
        # -- Context low-res --
        c_low = None
        if self.add_context_lowres:
            c_low = source_x
            # --- Normalize (X) ---
            if self.normalizer_x is not None:
                for i, variable in enumerate(self.vars_predictors):
                    normalizer_func = get_func_from_string(self.normalizer_x["module"], self.normalizer_x["normalizer_func_per_variable"][variable])
                    c_low[i,:] = normalizer_func(c_low[i,:], **self.normalizer_x["kwargs"][variable])
            # --- Transform to 2D (C_low) ---
            if self.transform_to_2D_x:
                C, G = c_low.shape
                c_low = c_low.reshape(C, self.H_x, self.W_x) # Shape (C, H, W)
            # print(f"c_low: {c_low.shape}")
            
        # -- Context high-res --
        c_high = None
        if self.add_pred_mean:
            c_high = source_r[num_vars:,:]
            # --- Transform to 2D (C_high) ---
            if self.transform_to_2D_y:
                C, G = c_high.shape
                c_high = c_high.reshape(C, self.H_y, self.W_y) # Shape (C, H, W)
            # print(f"c_high: {c_high.shape}")
            
        # -- Return --
        return residual, c_low, c_high





