## Load libraries
import os
import numpy as np
import torch
## Deep4production
from deep4production.core.trainers.trainer import trainer
from deep4production.core.pydatasets.pydataset_resdiff import pydataset_custom
##################################################################################################################################
class trainer_custom(trainer):
    """
    Custom trainer class for residual-based deep learning models.
    Purpose: Handles noise scheduling, regressor context, metadata updates, and batch training for residual denoising models.
    Parameters:
        data (dict): Dataset configuration.
        dataloader (dict): Dataloader parameters.
        output_dir (str): Output directory for experiment.
        model_info (dict): Model, loss, saving, and training parameters.
        graph (dict): Graph configuration for GNN models.
        d4dpy (dict): Custom pydataset configuration.
        Mlflow (dict): MLflow tracking configuration.
    """
    def __init__(self, data, dataloader, output_dir, model_info, graph, d4dpy, Mlflow):
        """
        Initializes the Residual Generator trainer.
        """
        ######### Call parent constructor to initialize common attributes #########
        super().__init__(
            data=data,
            dataloader=dataloader,
            output_dir=output_dir,
            model_info=model_info,
            graph=graph,
            d4dpy=d4dpy,
            Mlflow=Mlflow
        )

        # --- UPDATE SELF PARAMETERS ---------------------------------------
        self.noise_params = model_info["training_params"]["kwargs"]["noise_params"]
        self.path_regressor = d4dpy["kwargs"]["path_regressor"]
        self.add_pred_mean = d4dpy["kwargs"]["add_pred_mean"]
        self.add_context_lowres = d4dpy["kwargs"]["add_context_lowres"]

        # chh = self.model_params["kwargs"].get("cond_channels_high")
        # self.add_pred_mean = bool(chh)
        # chh = self.model_params["kwargs"].get("cond_channels_low")
        # self.add_context_lowres = bool(chh)
        print("📦 SELF UPDATED")

        # --- UPDATE METADATA ---------------------------------------
        self.update_metadata()


    # -------------------------------------------------------------------------
    def update_metadata(self):
        """
        Updates metadata dictionary with generator-specific parameters (noise, regressor, context).
        Purpose: Adds noise scheduling, regressor path, and context flags to metadata for reproducibility.
        Parameters:
            None (uses self attributes)
        Returns:
            None
        """
        ### Generator-specific metadata parameters
        self.metadata_dict["training_params"] = {}
        self.metadata_dict["training_params"]["noise_params"] = {k: v for k, v in self.noise_params.items()}
        self.metadata_dict["add_pred_mean"] = self.add_pred_mean
        self.metadata_dict["add_context_lowres"] = self.add_context_lowres
        self.metadata_dict["path_regressor"] = self.path_regressor
        ### Save metadata with the new information
        # self.save_metadata(self.metadata_path)

    # -------------------------------------------------------------------------
    def get_pydatasets(self):
        """
        Creates training and validation pydataset objects for residual models, updates metadata.
        Purpose: Instantiates pydataset objects, updates metadata, and prepares for training.
        Parameters:
            None (uses self attributes)
        Returns:
            tuple: (train_dataset, valid_dataset)
        """
        ## Create pydatasets
        kwargs_pydataset = {"predictors": self.data["predictors"], "predictands": self.data["predictands"], "load_in_memory": self.data.get("load_in_memory", True)}
        kwargs_pydataset.update(**self.d4dpy)
        kwargs_pydataset.update({"dataset": "training"})
        train_dataset = self.pydataset(temporal_period = self.data["training_period"], **kwargs_pydataset)
        valid_dataset = None
        if self.data.get("validation_period", None) is not None:
            kwargs_pydataset.update({"dataset": "validation"})
            valid_dataset = self.pydataset(temporal_period = self.data["validation_period"], **kwargs_pydataset)
        ### Update metadata and save it with the new information
        self.metadata_dict = self.cont_metadata(train_dataset) 
        # self.save_metadata(self.metadata_path)
        print("📦 PYDATASETS READY")
        return train_dataset, valid_dataset
        
    # -------------------------------------------------------------------------
    def sigma(self, P_mean, P_std, sigma_min, sigma_max, batch_size):
        """
        Samples noise schedule for diffusion models using log-normal distribution.
        Purpose: Generates noise scaling factors for each batch.
        Parameters:
            P_mean (torch.Tensor): Mean parameter for log-normal noise.
            P_std (torch.Tensor): Std parameter for log-normal noise.
            sigma_min (float): Minimum noise value.
            sigma_max (float): Maximum noise value.
            batch_size (int): Batch size.
        Returns:
            torch.Tensor: Noise schedule tensor.
        """
        z = torch.randn(batch_size, 1, 1, 1)  # standard normal
        sigma_t = torch.exp(P_mean + P_std * z)
        sigma_t = sigma_t.clamp(min=sigma_min, max=sigma_max) 
        return sigma_t

    # -------------------------------------------------------------------------
    def model_backprop(self, model, data, optimizer, loss_function, device, noise_params, is_this_training=True):
        """
        Performs a single forward and backward pass for a batch, including noise scheduling and conditioning.
        Purpose: Handles core training step for residual denoising models, including noise injection and loss computation.
        Parameters:
            model: PyTorch model.
            data: Tuple of residual, context low-res, context high-res arrays.
            optimizer: PyTorch optimizer.
            loss_function: Loss function callable.
            device: Device string ('cpu' or 'cuda').
            noise_params (dict): Noise scheduling parameters.
            is_this_training (bool): Whether to perform backpropagation.
        Returns:
            float: Loss value for the batch.
        """
        # --- Get noise scheduler parameters ---
        P_mean, P_std, sigma_min, sigma_max, sigma_data = noise_params["P_mean"], noise_params["P_std"], noise_params["sigma_min"], noise_params["sigma_max"], noise_params["sigma_data"]

        # --- Get arrays as defined in the pydataset class. ---
        r, c_low, c_high = data

        # --- Get batch size ---
        batch_size = r.shape[0]
    
        # --- Condition features and target ---
        r = r.to(device)
        # print(f"r {r.shape}")
        c_high = c_high.to(device)
        # print(f"c_high {c_high.shape}")
        c_low = c_low.to(device)
        # print(f"c_low {c_low.shape}")

        # --- Sample noise and forward pass ---
        sigma_t = self.sigma(P_mean, P_std, sigma_min=sigma_min, sigma_max=sigma_max, batch_size=batch_size).to(device)
        # print(f"sigma_t {sigma_t.shape}")
        z = torch.randn_like(r)  # ε ~ N(0, I)
        r_t = r + sigma_t * z  # x_t = x_0 + σ_t * ε
        # print(f"r_t {r_t.shape}")

        # --- Compute conditioning scalars ---
        c_in = 1.0 / torch.sqrt(sigma_data ** 2 + sigma_t ** 2)
        # c_noise = 0.25 * torch.log(sigma_t)
        # print(f"c_noise {c_noise.shape}")
        r_t_scaled = c_in * r_t
        r_t_scaled = r_t_scaled.to(device)
        # print(f"r_t_scaled {r_t_scaled.shape}")

        # --- Feed features to the denoiser deep learning model ---
        optimizer.zero_grad()
        denoiser_output = model(x=r_t_scaled, context_low=c_low, context_high=c_high)

        # --- Compute loss ---
        loss = loss_function(target=r, output=denoiser_output, sigma_t=sigma_t, r_t=r_t)
        if is_this_training:
            loss.backward()

        # --- Return ---
        return loss.item()
