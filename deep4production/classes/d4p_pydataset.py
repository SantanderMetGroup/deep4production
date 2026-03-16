import zarr
import yaml
import importlib
import numpy as np
import xarray as xr 
import pandas as pd
from torch import from_numpy
from torch.utils.data import Dataset
import torch
## Deep4production
from deep4production.utils.trans import from_pred_to_xarray
from deep4production.utils.normalizers import d4dnormalizers
from deep4production.utils.general import get_func_from_string
from deep4production.utils.temporal import get_dates_from_yaml, get_sample_map, get_pairs
########################################################################################################
########################################################################################################
class d4p_pydataset(Dataset):
    """
    Dataset class for loading, preprocessing, and batching predictor/predictand/forcing data for deep learning.
    Purpose: Handles variable selection, normalization, operator application, temporal alignment, and batching for PyTorch models.
    Parameters:
        predictors (dict): Predictor dataset configuration.
        predictands (dict): Predictand dataset configuration.
        temporal_period (list): List of target dates.
        load_in_memory (bool): Whether to load all data into memory.
        forcings (dict, optional): Forcing dataset configuration.
    """
    def __init__(self, predictors: dict, predictands: dict, temporal_period: list, load_in_memory: bool = True, forcings={}): 
        # --- Parameters (X, Y) --- 
        path_predictors, path_predictands = predictors["paths"], predictands["paths"]
        variables_predictors, variables_predictands, variables_forcings = predictors.get("variables", None), predictands.get("variables", None), forcings.get("variables", None)
        normalizer_predictors, normalizer_predictands, normalizer_forcings = predictors.get("normalizer", None), predictands.get("normalizer", None), forcings.get("normalizer", None)
        operator_predictors, operator_predictands, operator_forcings = predictors.get("operator", None), predictands.get("operator", None), forcings.get("operator", None)
        self.transform_to_2D_x, self.transform_to_2D_y = predictors.get("transform_to_2D", False), predictands.get("transform_to_2D", False)
        self.num_lagged_x, self.num_lagged_y = predictors.get("num_lagged", 0), predictands.get("num_lagged", 0)

        # --- Load metadata ---
        self.x, self.vars_x, self.idx_vars_x, self.normalizer_x, self.operator_x, self.H_x, self.W_x, self.G_x = self.get_data_info(path_predictors, variables_predictors, normalizer_predictors, operator_predictors)
        self.y, self.vars_y, self.idx_vars_y, self.normalizer_y, self.operator_y, self.H_y, self.W_y, self.G_y = self.get_data_info(path_predictands, variables_predictands, normalizer_predictands, operator_predictands)
        self.forcings = forcings
        if forcings:
            _, self.vars_f, self.idx_vars_f, self.normalizer_f, self.operator_f, __, ___, _____ = self.get_data_info(path_predictands, variables_forcings, normalizer_forcings, operator_forcings)
        else:
            self.vars_f = None
            self.idx_vars_f = None
            self.normalizer_f = None
            self.operator_f = None

        # --- Temporal information (intersect X and Y and get indexing info)--- 
        freq = self.x[0].attrs.get("temporal_freq")
        dates_yaml = get_dates_from_yaml(temporal_period, freq=freq)
        self.sample_map_x, dates_x = get_sample_map(dates_yaml, self.x)
        self.sample_map_y, dates_y = get_sample_map(dates_yaml, self.y)
        dates = sorted(set(dates_x) & set(dates_y))
        self.pairs = get_pairs(dates=dates, freq=freq, num_lagged_x=self.num_lagged_x)
        self.target_dates = list(self.pairs.keys())
        self.num_samples = len(self.pairs)
        print(f"📊 Number of samples: {self.num_samples}")
        if self.num_samples == 0:
            assert f"❌ There are no common dates between the predictor (X) and predictand (Y) datasets."

        # --- Load in memory? ---
        if load_in_memory: # If dataset fits in memory, load all predictors to speed up
            x_data = [np.array(x) for x in self.x]
            y_data = [np.array(y) for y in self.y]
            self.data = {"x": x_data, "y": y_data}
            print("📦 DATA LOADED INTO MEMORY FOR FASTER ACCESS")
        else:
            self.data = {"x": self.x, "y": self.y}
        
    # -------------------------------------------------------------------------
    def get_data_info(self, path_data, variables, normalizer_info, operator_info):
        """
        Loads metadata and variable info from Zarr files, including normalizer and operator setup.
        Parameters:
            path_data (list): List of Zarr file paths.
            variables (list): Variable names.
            normalizer_info (dict): Normalizer configuration.
            operator_info (dict): Operator configuration.
        Returns:
            tuple: (files, vars, idx_vars, normalizer, operator, H, W, G)
        """
        # --- Files ---
        files = [zarr.open(p, mode='r') for p in path_data]
        # --- Variables ---
        print("⚠️  WARNING: For subsetting variables from zarr it is assumed that the order of variables is the same across zarr files, e.g., original order of variables in self.x[0] is the same that self.x[1]")
        if variables is None:  # Selecting all available variables in the dataset
            vars = [var for var, idx in files[0].attrs["variables"].items()]
            idx_vars = [idx for var, idx in files[0].attrs["variables"].items()]
        else:
            vars = variables
            idx_vars = [files[0].attrs["variables"][var] for var in vars if var in files[0].attrs["variables"] ]
        # --- Normalizer ---
        normalizer = None
        if normalizer_info is not None:
            normalizer = {}
            normalizer_info_default = normalizer_info.get("default", None)
            normalizer["dataset"] = normalizer_info["path_reference"]
            normalizer["kwargs"] = {var: self.get_statistics_from_zarr_file(normalizer["dataset"], var = var) for var in vars}
            normalizer["normalizer_func_per_variable"] = {var: (normalizer_info[var] if var in normalizer_info else normalizer_info_default) for var in vars}
            print(f"--- Normalizer for variables: {vars} ---")
            print(normalizer["normalizer_func_per_variable"])
        # --- Operator ---
        operator = None
        if operator_info is not None:
            operator = {}
            operator_info_default = operator_info.get("default", None)
            operator["module"] = "deep4production.utils.operators"
            operator["operator_func_per_variable"] = {var: (operator_info[var] if var in operator_info else operator_info_default) for var in vars}
            print(f"--- Operator for variables: {vars} ---")
            print(operator["operator_func_per_variable"])
        # --- Height and width (H and W) ---
        H, W = files[0].attrs.get("H", None), files[0].attrs.get("W", None)
        # --- Number of gridpoints (G) ---
        G = files[0].attrs.get("shape")[2]
        # --- Return ---
        return files, vars, idx_vars, normalizer, operator, H, W, G

    # -------------------------------------------------------------------------
    def get_forcings_info(self):
        """
        Returns information about forcings variables, indices, normalizer, and operator.
        Returns:
            tuple: (vars_f, idx_vars_f, normalizer_f, operator_f)
        """
        return self.vars_f, self.idx_vars_f, self.normalizer_f, self.operator_f

    # -------------------------------------------------------------------------
    def get_coords(self):
        """
        Returns latitude and longitude arrays for predictands.
        Returns:
            tuple: (lats, lons)
        """
        lats = np.array(self.y[0].attrs.get("lats"), dtype=np.float32)
        lons = np.array(self.y[0].attrs.get("lons"), dtype=np.float32)
        return lats, lons

    # -------------------------------------------------------------------------
    def get_spatial_dims(self):
        """
        Returns spatial dimensions (height, width) for predictors and predictands.
        Returns:
            tuple: (H_x, W_x, H_y, W_y)
        """
        return self.H_x, self.W_x, self.H_y, self.W_y

    # -------------------------------------------------------------------------
    def get_vars(self):
        """
        Returns variable names for predictors and predictands.
        Returns:
            tuple: (vars_x, vars_y)
        """
        return self.vars_x, self.vars_y

    # -------------------------------------------------------------------------
    def get_num_gridpoints(self):
        """
        Returns number of gridpoints for predictors and predictands.
        Returns:
            tuple: (G_x, G_y)
        """
        return self.G_x, self.G_y

    # -------------------------------------------------------------------------
    def get_transform2D(self):
        """
        Returns transform-to-2D flags for predictors and predictands.
        Returns:
            tuple: (transform_to_2D_x, transform_to_2D_y)
        """
        return self.transform_to_2D_x, self.transform_to_2D_y

    # -------------------------------------------------------------------------
    def get_lagged_info(self):
        """
        Returns number of lagged timesteps for predictors and predictands.
        Returns:
            tuple: (num_lagged_x, num_lagged_y)
        """
        return self.num_lagged_x, self.num_lagged_y

    # -------------------------------------------------------------------------
    def get_normalizer_info(self, predictands=False):
        """
        Returns normalizer info for predictors or predictands.
        Parameters:
            predictands (bool): If True, returns for predictands; else for predictors.
        Returns:
            dict or None: Normalizer info.
        """
        if predictands:
            return self.normalizer_y
        else:
            return self.normalizer_x

    # -------------------------------------------------------------------------
    def get_operator_info(self, predictands=False):
        """
        Returns operator info for predictors or predictands.
        Parameters:
            predictands (bool): If True, returns for predictands; else for predictors.
        Returns:
            dict or None: Operator info.
        """
        if predictands:
            return self.operator_y
        else:
            return self.operator_x

    # -------------------------------------------------------------------------
    def get_target_samples(self):
        """
        Returns target samples as xarray.Dataset for all dates in the dataset.
        Returns:
            xarray.Dataset: Target samples stacked along time.
        """
        target_samples = []
        # --- Loop over samples ---
        for idx in range(len(self)):
            # --- Get dates ---
            target_date = self.target_dates[idx] 
            dates = self.pairs[target_date]
            # --- Prepare target ---
            y = self.preprocess(target_date, self.data["y"], self.vars_y, self.idx_vars_y, self.sample_map_y, operator=None, normalizer=None, transform_to_2D=None, H=None, W=None).unsqueeze(0).cpu().numpy() # Add time dimension
            # --- To xarray ---
            ds = from_pred_to_xarray(data_pred=y,time_pred=np.datetime64(target_date), vars=self.vars_y, lats=self.y[0].attrs["lats"], lons=self.y[0].attrs["lons"])
            target_samples.append(ds)
        # --- Stack ---
        target_samples = xr.concat(target_samples, dim="time")
        return target_samples

    # -------------------------------------------------------------------------
    def get_statistics_from_zarr_file(self, zarr_file_path, var, axis=0):
        """
        Extracts mean, std, min, max statistics from a Zarr file for a variable.
        Parameters:
            zarr_file_path (str): Path to Zarr file.
            var (str): Variable name.
            axis (int): Axis for broadcasting.
        Returns:
            dict: Statistics dictionary.
        """
        zarr_file = zarr.open(zarr_file_path, mode='r')
        # Build stats dictionary
        mean_val = np.array(zarr_file.attrs['mean'][var], dtype=np.float32)
        std_val = np.array(zarr_file.attrs['std'][var], dtype=np.float32)
        min_val = np.array(zarr_file.attrs['min'][var], dtype=np.float32)
        max_val = np.array(zarr_file.attrs['max'][var], dtype=np.float32)
        kwargs_normalizer = {"mean": mean_val, "std": std_val, "min": min_val, "max": max_val}
        # Return
        return kwargs_normalizer

    # -------------------------------------------------------------------------
    def preprocess(self, date, data, vars, idx_vars, sample_map, operator=None, normalizer=None, transform_to_2D=False, H=None, W=None):
        """
        Preprocesses a sample: indexing, operator, normalization, reshaping, and conversion to torch tensor.
        Parameters:
            date: Target date.
            data: Data array.
            vars: Variable names.
            idx_vars: Variable indices.
            sample_map: Sample mapping.
            operator: Operator info (optional).
            normalizer: Normalizer info (optional).
            transform_to_2D (bool): Whether to reshape to 2D.
            H, W: Height and width for reshaping.
        Returns:
            torch.Tensor: Preprocessed sample.
        """
        # -- Get sample --
        i, j = sample_map[date]
        source = data[i][j]
        x = source[idx_vars] # Shape (C, G)
        # --- Operator ---  
        if operator is not None:
            for c, variable in enumerate(vars):
                if operator["operator_func_per_variable"][variable] is not None:
                    operator_func = get_func_from_string(operator["module"], operator["operator_func_per_variable"][variable])
                    x[c,:] = operator_func(x[c,:])
        # --- Normalize ---  
        if normalizer is not None:
            for c, variable in enumerate(vars):
                if normalizer["normalizer_func_per_variable"][variable] is not None:
                    normalizer_class = d4dnormalizers(**normalizer["kwargs"][variable])
                    normalizer_method = getattr(normalizer_class, normalizer["normalizer_func_per_variable"][variable])
                    x[c,:] = normalizer_method(x[c,:])
        # --- Transform to 2D ---
        if transform_to_2D:
            C, G = x.shape
            x = x.reshape(C, H, W) # Shape (C, H, W)
        # --- Convert to torch tensor ---
        x = from_numpy(x)
        # --- Return ---  
        return x

    # -------------------------------------------------------------------------
    def __len__(self):
        """
        Returns number of samples in the dataset.
        Returns:
            int: Number of samples.
        """
        return self.num_samples

    # -------------------------------------------------------------------------
    def __getitem__(self, idx):
        """
        Returns a tuple (x, y, f) for a given sample index.
        Parameters:
            idx (int): Sample index.
        Returns:
            tuple: (x, y, f)
        """
        # --- Prepare data ---
        target_date = self.target_dates[idx] 
        dates = self.pairs[target_date]
        # ---
        if len(dates) > 1:
            x = []
            for date in dates:
                x.append(self.preprocess(date, self.data["x"], self.vars_x, self.idx_vars_x, self.sample_map_x, operator=self.operator_x, normalizer=self.normalizer_x, transform_to_2D=self.transform_to_2D_x, H=self.H_x, W=self.W_x))
            x = torch.stack(x)
        else:
            x = self.preprocess(target_date, self.data["x"], self.vars_x, self.idx_vars_x, self.sample_map_x, operator=self.operator_x, normalizer=self.normalizer_x, transform_to_2D=self.transform_to_2D_x, H=self.H_x, W=self.W_x)
        # print(f"x shape: {x.shape}")      
        # ---
        y = self.preprocess(target_date, self.data["y"], self.vars_y, self.idx_vars_y, self.sample_map_y, operator=self.operator_y, normalizer=self.normalizer_y, transform_to_2D=self.transform_to_2D_y, H=self.H_y, W=self.W_y)
        # print(f"y shape: {y.shape}")
        # --- Forcings (f) ---
        if self.forcings:
            f = self.preprocess(target_date, self.data["y"], self.vars_f, self.idx_vars_f, self.sample_map_y, operator=self.operator_f, normalizer=self.normalizer_f, transform_to_2D=self.transform_to_2D_y, H=self.H_y, W=self.W_y)
            # print(f"f shape: {f.shape}")
        else:
            f = "N/A"
        # --- Return ---
        return x, y, f