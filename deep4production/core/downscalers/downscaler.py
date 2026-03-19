import zarr
import torch
import numpy as np
import xarray as xr
from torch import from_numpy
## Deep4production
from deep4production.deep.utils import load_model
from deep4production.utils.trans import from_pred_to_xarray
from deep4production.utils.normalizers import d4pnormalizers
from deep4production.utils.general import get_func_from_string
from deep4production.utils.temporal import get_dates_from_yaml, get_sample_map, get_pairs

##################################################################################################################################
class downscaler:
    """
    Downscaler class for applying trained models to input data and generating predictions.
    Purpose: Loads model and metadata, preprocesses input, handles forcings, and saves predictions.
    Parameters:
        id_dir (str): Experiment directory.
        input_data (dict): Input data configuration.
        model_file (str, optional): Model checkpoint filename.
        saving_info (dict, optional): Output saving configuration.
        ensemble_size (int, optional): Number of ensemble members.
        graph (dict, optional): Graph configuration for GNN models.
        forcing_data (dict, optional): Forcing data configuration.
    """
    def __init__(self, id_dir, input_data, model_file=None, saving_info=None, ensemble_size=1, graph=None, forcing_data=None):
        """
        Initializes the D4P Downscaler.
        """
        print("🚀 STARTING D4P DOWNSCALER")
        # --- SELF PARAMETERS ---
        self.ensemble_size = ensemble_size
        self.graph = graph
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        print("📦 SELF READY")

        # --- GET MODEL AND METADATA FROM CHECKPOINT ---
        if model_file is not None:
            model_path = f"{id_dir}/models/{model_file}"
            self.model, self.metadata = load_model(path=model_path, map_location=self.device, return_metadata=True)
            self.model.to(self.device)
            print("📦 MODEL AND METADATA LOADED")

        # --- OUTPUT PATH ---
        self.saving_info = saving_info 
        if self.saving_info is not None:
            file = self.saving_info["file"]
            self.output_path = f"{id_dir}/predictions/{file}"
            print(f"📦 PREDICTIONS WILL BE SAVED HERE: {self.output_path}")

        # --- GET INFO FROM METADATA ---
        self.update_self(input_data["paths"])
        print("📦 SELF UPDATED WITH INFO FROM CHECKPOINT'S METADATA")

        # --- GET DOWNSCALING DATES ---
        freq = self.x[0].attrs["temporal_freq"]
        dates_yaml = get_dates_from_yaml(input_data["years"], freq=freq)
        self.sample_map, dates = get_sample_map(dates_yaml, self.x)
        self.pairs = get_pairs(dates=dates, freq=freq, num_lagged_x=self.num_lagged_x)
        self.target_dates = list(self.pairs.keys())
        num_samples = len(self.pairs)
        print(f"📊 Number of initialization dates: {num_samples}")

        # --- LOAD INPUT DATA IN MEMORY? ---
        load_in_memory = input_data.get("load_in_memory", True)
        if load_in_memory: # If dataset fits in memory, load input data to speed up
            x_data = [np.array(x) for x in self.x]
            self.data = {"x": x_data}
            print("📦 DATA LOADED INTO MEMORY FOR FASTER ACCESS")
        else:
            self.data = {"x": self.x}
        
        # --- FORCINGS (optional) ---
        self.forcing_data = forcing_data
        if self.forcing_data is not None:
            self.update_self_with_forcings(forcing_data["paths"])
            freq = self.f[0].attrs["temporal_freq"]
            dates_yaml = get_dates_from_yaml(forcing_data["years"], freq=freq)
            self.sample_map_f, _ = get_sample_map(dates_yaml, self.x)
            load_in_memory = input_data.get("load_in_memory", True)
            if load_in_memory: # If dataset fits in memory, load input data to speed up
                f_data = [np.array(f) for f in self.f]
                self.data.update({"f": f_data})
                print("📦 FORCING DATA LOADED INTO MEMORY FOR FASTER ACCESS")
            else:
                self.data.update({"f": self.f})

        # --- MAPPING TO XARRAY INFO ---
        ## Template
        if self.saving_info is not None:
            template_path = self.saving_info.get("template", None)
            if template_path is not None:
                self.template = xr.open_dataset(template_path)
                self.lats = None
                self.lons = None
                print("✅ Using provided template for coordinates.")
            else:
                self.template = None
                self.lats = self.metadata["lats_y"]
                self.lons = self.metadata["lons_y"]
                print("⚠️ No template provided. Using lats and lons from metadata.")
        else:
            self.template = None
            self.lats = self.metadata["lats_y"]
            self.lons = self.metadata["lons_y"]
            print("⚠️ No template provided. Using lats and lons from metadata.")
        ## Formatting
        self.format_output = None
        if self.saving_info is not None:
            self.format_output = self.saving_info.get("formatting", None)
            if self.format_output is not None:
                formatting_module = "deep4production.utils.formatting"
                formatting_name = self.saving_info["formatting"]["name"]
                self.formatting_func = get_func_from_string(formatting_module, formatting_name)
                self.formatting_kwargs = self.saving_info["formatting"].get("kwargs", None)

        # --- BUILD GRAPH ---------------------------------------
        if self.graph is not None:
            if self.graph["path"] is not None:
                self.edge_index = torch.load(f"{id_dir}/aux_files/{self.graph['path']}", weights_only=False)
                print(f"📦 GRAPH LOADED FROM: {self.graph['path']}")
            else:
                self.edge_index = get_func_from_string(module_string=self.graph["module"],func_string=self.graph["name"], kwargs=self.graph.get("kwargs", None))
                torch.save(self.edge_index, f"{self.aux_dir}/aux_files/edge_index_B.pt")
                print(f"📦 GRAPH READY: function {self.graph['name']} from {self.graph['module']}")

        # --- POSTPROCESS FUNC ---------------------------------------
        postprocess_module = "deep4production.deep.postprocessors"
        # Specific
        self.post_func_kwargs = {}
        if self.loss_params["name"] == "NLLBerGammaLoss":
            postprocess_name = "from_bergamma_to_pred"
            self.post_func_kwargs = {"threshold": self.loss_params["kwargs"]["threshold"]} 
        elif self.loss_params["name"] == "NLLGaussianLoss":
            postprocess_name = "from_gaussian_to_pred"
        else:
            postprocess_name = "standard"
        self.post_func = get_func_from_string(postprocess_module, postprocess_name)

    # ---------------------------------------------------------------------------------------------------------------------<
    def update_self(self, paths):
        """
        Updates internal attributes using input paths and metadata.
        Parameters:
            paths (list): List of Zarr file paths.
        """
        # --- Files (X)---
        self.x = [zarr.open(p, mode='r') for p in paths]
        # --- Variables --- 
        self.vars_y = self.metadata["vars_y"]
        self.vars_x = self.metadata["vars_x"]
        self.idx_vars_x = [self.x[0].attrs["variables"][var] for var in self.vars_x]
        self.num_lagged_x = self.metadata["num_lagged_x"]
        # --- Normalizer ---
        self.normalizer_x = self.metadata.get("normalizer_x", None)
        if self.normalizer_x is not None:
            print("--- Normalizer (X) ---")
            print(self.normalizer_x.get("normalizer_func_per_variable", None))
        # --- Denormalizer (Prediction) ---
        self.normalizer_y = self.metadata.get("normalizer_y", None)
        if self.normalizer_y is not None:
            print("--- Denormalizer (Y) ---")
            print(self.normalizer_y.get("normalizer_func_per_variable", None))
        # --- Operator ---
        self.operator_x = self.metadata.get("operator_x", None)
        if self.operator_x is not None:
            print("--- Operator (X) ---")
            print(self.operator_x.get("operator_func_per_variable", None))
        # --- Deoperator (Prediction) ---
        self.operator_y = self.metadata.get("operator_y", None)
        if self.operator_y is not None:
            print("--- Deoperator (Y) ---")
            print(self.operator_y.get("operator_func_per_variable", None))
        # --- Loss params --- 
        self.loss_params = self.metadata.get("loss_params", None)
        # --- Transform to 2D --- 
        self.transform_to_2D_x = self.metadata.get("transform_to_2D_x", False)
        self.transform_to_2D_y = self.metadata.get("transform_to_2D_y", False)
        # --- Input and output 2D spatial dimensions ---
        self.H_x, self.W_x = self.metadata.get("H_x", None), self.metadata.get("W_x", None)
        self.H_y, self.W_y = self.metadata.get("H_y", None), self.metadata.get("W_y", None)
        # --- Input and output expected number of gridpoints ---
        self.G_x = self.metadata.get("G_x", None)
        self.G_y = self.metadata.get("G_y", None)


    # ---------------------------------------------------------------------------------------------------------------------<
    def update_self_with_forcings(self, fpaths):
        """
        Updates internal attributes for forcings using input paths and metadata.
        Parameters:
            fpaths (list): List of Zarr file paths for forcings.
        """
        # --- Forcings info ---
        self.f = [zarr.open(p, mode='r') for p in fpaths]
        self.vars_f = self.metadata["vars_f"]
        self.idx_vars_f = [self.f[0].attrs["variables"][var] for var in self.vars_f]
        self.normalizer_f = self.metadata.get("normalizer_f", None)
        if self.normalizer_f is not None:
            print("--- Normalizer (F) ---")
            print(self.normalizer_f.get("normalizer_func_per_variable", None))
        self.operator_f = self.metadata.get("operator_f", None)
        if self.operator_f is not None:
            print("--- Operator (F) ---")
            print(self.operator_f.get("operator_func_per_variable", None))


    # ---------------------------------------------------------------------------------------------------------------------<
    def graphPredict(self, x, edge_index, model, f=["N/A"]):
        """
        Placeholder for graph prediction. Should be implemented in subclass for PyTorch Geometric models.
        Parameters:
            x (torch.Tensor): Input tensor.
            edge_index: Graph edge indices.
            model: PyTorch model.
            f: Forcing tensor (optional).
        Returns:
            np.ndarray: Prediction array.
        """
        assert False, (
            "🛑 Placeholder for the graphPredict function. Create a subclass of d4p_downscaler "
            "that implements graphPredict to convert the PyTorch data into a format compatible "
            "with PyTorch Geometric (PyG) graph objects."
        )

    # ---------------------------------------------------------------------------------------------------------------------<
    def preprocess(self, date, data, vars, idx_vars, sample_map, operator=None, normalizer=None, transform_to_2D=False, H=None, W=None):
        """
        Preprocesses a sample for model input: indexing, operator, normalization, reshaping, and conversion to tensor.
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
                    normalizer_class = d4pnormalizers(**normalizer["kwargs"][variable])
                    normalizer_method = getattr(normalizer_class, normalizer["normalizer_func_per_variable"][variable])
                    x[c,:] = normalizer_method(x[c,:])
        # --- Transform to 2D ---
        if transform_to_2D:
            C, G = x.shape
            x = x.reshape(C, H, W) # Shape (C, H, W)
        # --- Convert to torch tensor ---
        x = from_numpy(x.copy())
        # --- Return ---  
        return x.to(self.device)  # Shape (B, C, ...)

    # ---------------------------------------------------------------------------------------------------------------------<
    def postprocess(self, date, data, vars, member, operator=None, normalizer=None, lats=None, lons=None, template=None, func=None, kwargs=None):
        """
        Postprocesses model output: denormalization, deoperator, formatting, and conversion to xarray.
        Parameters:
            date: Target date.
            data: Prediction array.
            vars: Variable names.
            member: Ensemble member index.
            operator: Operator info (optional).
            normalizer: Normalizer info (optional).
            lats, lons: Latitude and longitude arrays.
            template: xarray template (optional).
            func: Postprocessing function (optional).
            kwargs: Additional arguments for postprocessing.
        Returns:
            xarray.Dataset: Prediction in xarray format.
        """
        # --- De-transform from 2D? ---
        if self.transform_to_2D_y:
            B, C, H, W = data.shape
            data = data.reshape(B, C, H*W) # Shape (B, C, G)

        # -- FUNC -- 
        if func is not None:
            # print(f"BEFORE: {data.shape}")
            data = func(data, **kwargs)
            # print(f"AFTER: {data.shape}")
        # -- Denormalize --
        if normalizer is not None:
            for c, variable in enumerate(vars):
                if normalizer["normalizer_func_per_variable"][variable] is not None:
                    normalizer_class = d4pnormalizers(**normalizer["kwargs"][variable])
                    normalizer_method = getattr(normalizer_class, normalizer["normalizer_func_per_variable"][variable])
                    data[:,c,:] = normalizer_method(data[:,c,:], denormalize=True)
        # --- Deoperator ---  
        if operator is not None:
            for c, variable in enumerate(vars):
                if operator["operator_func_per_variable"][variable] is not None:
                    operator_func = get_func_from_string(operator["module"], operator["operator_func_per_variable"][variable])
                    data[:,c,:] = operator_func(data[:,c,:], back=True)
        # -- Prediction to xarray --
        date = np.datetime64(date)
        ds_pred = from_pred_to_xarray(data, date, vars, lats, lons, template, self.H_y, self.W_y)
        # -- Return (Prediction in xarray format) --
        ds_pred = ds_pred.assign_coords({"member": member})
        return ds_pred

    # ---------------------------------------------------------------------------------------------------------------------<
    def downscale(self, model=None, return_pred=False, verbose=True):
        """
        Runs the downscaling process: preprocesses input, predicts, postprocesses, and saves or returns output.
        Parameters:
            model: PyTorch model (optional).
            return_pred (bool): Whether to return prediction instead of saving.
            verbose (bool): Print progress messages.
        Returns:
            xarray.Dataset or None: Prediction dataset if return_pred is True, otherwise saves the prediction file.
        """
        if verbose:
            print("🚀 STARTING DOWNSCALING PROCESS")
        # --- Get model ---
        if model is None:
            model = self.model
        # --- Loop over dates ---
        ds_out = []
        for member in range(self.ensemble_size):
            ps = []
            for target_date in self.target_dates:  
                if verbose:  
                    print(f"📅 Member: {member+1}/{self.ensemble_size}. Downscaling date: {target_date}")
                dates = self.pairs[target_date]

                # -- Preprocess (indexing, normalizing,..) --
                if len(dates) > 1:
                    inp = []
                    for date in dates:
                        inp.append(self.preprocess(date, self.data["x"], self.vars_x, self.idx_vars_x, self.sample_map, operator=self.operator_x, normalizer=self.normalizer_x, transform_to_2D=self.transform_to_2D_x, H=self.H_x, W=self.W_x))
                    inp = torch.stack(inp).unsqueeze(0)
                else:
                    inp = self.preprocess(target_date, self.data["x"], self.vars_x, self.idx_vars_x, self.sample_map, operator=self.operator_x, normalizer=self.normalizer_x, transform_to_2D=self.transform_to_2D_x, H=self.H_x, W=self.W_x).unsqueeze(0)
                # print(f"Inp shape: {inp.shape}")

                # -- High-res forcings (indexing, normalizing,..) --
                if self.forcing_data is not None:
                    f = self.preprocess(target_date, self.data["f"], self.vars_f, self.idx_vars_f, self.sample_map_f, operator=self.operator_f, normalizer=self.normalizer_f, transform_to_2D=self.transform_to_2D_y, H=self.H_y, W=self.W_y).unsqueeze(0)
                else:
                    Cy = len(self.vars_y)
                    spatial = [self.H_y, self.W_y] if self.transform_to_2D_y else [self.G_y]
                    f = torch.zeros(1, Cy, *spatial, device=self.device)
                # print(f"F shape: {f.shape}")

                # -- Predict --
                if self.graph is not None:
                    p_torch = self.graphPredict(x=inp, edge_index=self.edge_index, model=model, f=f)
                else:
                    with torch.no_grad():
                        p_torch = model(inp, f)
                # print(f"Pred shape: {p.shape}")
                p = p_torch.cpu().numpy()                        
                del inp, f, p_torch

                # -- Postprocess (denormalizing, xarray formatting,..) --
                p = self.postprocess(date=target_date, data=p, vars=self.vars_y, member=member, operator=self.operator_y, normalizer=self.normalizer_y, lats=self.lats, lons=self.lons, template=self.template, func=self.post_func, kwargs=self.post_func_kwargs)
                ps.append(p)
            # --- Merge predictions along time ---
            ds_member = xr.concat(ps, dim="time")
            ds_out.append(ds_member)
        # --- Merge predictions along member ---
        ds_out = xr.concat(ds_out, dim="member")
        # --- Format output ---
        if self.format_output:
            ds_out = self.formatting_func(ds_out, **self.formatting_kwargs)
        # --- Save to disk or return prediction ---
        if return_pred:
            return ds_out
        print(ds_out)
        ds_out.to_netcdf(self.output_path)


