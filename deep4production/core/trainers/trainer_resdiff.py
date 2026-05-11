## Load libraries
import os
import numpy as np
import torch
## Deep4production
from deep4production.core.trainers.trainer import trainer
from deep4production.core.pydatasets.pydataset_resdiff import pydataset_custom
from deep4production.utils.log import get_logger

log = get_logger("trainer.resdiff")
##################################################################################################################################
class trainer_custom(trainer):
    """
    Custom trainer class for residual-based deep learning models.
    Purpose: Handles noise scheduling, regressor context, metadata updates, and batch training for residual denoising models.
    Parameters:
        data (dict): Dataset configuration.
        dataloader (dict): Dataloader parameters.
        id_dir (str): Directory for experiment outputs.
        model_info (dict): Model, loss, saving, and training parameters.
        graph (dict): Graph configuration for GNN models.
        d4dpy (dict): Custom pydataset configuration.
        Mlflow (dict): MLflow tracking configuration.
    """
    def __init__(self, data, dataloader, id_dir, model_info, graph, d4dpy, Mlflow,
                 normalizer_info_x=None, normalizer_info_y=None, normalizer_info_f=None,
                 hardware=None):
        """
        Initializes the Residual Generator trainer.
        """
        ######### Call parent constructor to initialize common attributes #########
        super().__init__(
            data=data,
            dataloader=dataloader,
            id_dir=id_dir,
            model_info=model_info,
            graph=graph,
            d4dpy=d4dpy,
            Mlflow=Mlflow,
            normalizer_info_x=normalizer_info_x,
            normalizer_info_y=normalizer_info_y,
            normalizer_info_f=normalizer_info_f,
            hardware=hardware,
        )

        # --- UPDATE SELF PARAMETERS ---------------------------------------
        self.noise_params = model_info["training_params"]["kwargs"]["noise_params"]
        self.path_regressor = d4dpy["kwargs"]["path_regressor"]
        self.add_pred_mean = d4dpy["kwargs"]["add_pred_mean"]
        self.add_context_lowres = d4dpy["kwargs"]["add_context_lowres"]

        log.debug("ResDiff trainer self-update complete")

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
        # setdefault avoids clobbering anything the base trainer may put under
        # training_params in the future; canonical noise_params path is
        # metadata.training_params.noise_params (shared with trainer_cpmgem).
        self.metadata_dict.setdefault("training_params", {})["noise_params"] = {k: v for k, v in self.noise_params.items()}
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
        kwargs_pydataset = {"predictors": self.data["predictors"], "predictands": self.data["predictands"], "load_in_memory": self.data.get("load_in_memory", True), "cache_mb": self.data.get("zarr_cache_mb", None)}
        kwargs_pydataset.update(**self.d4dpy)
        # The resdiff pydataset needs to normalize x and y on CPU for the
        # one-shot residuals precomputation (the regressor and its training
        # data both live in normalized space). Pass the recipe dicts down so
        # it can build its own local InputNormalizer instances.
        kwargs_pydataset["normalizer_info_x"] = self.normalizer_info_x
        kwargs_pydataset["normalizer_info_y"] = self.normalizer_info_y
        kwargs_pydataset.update({"dataset": "training"})
        train_dataset = self.pydataset(temporal_period = self.data["training_period"], **kwargs_pydataset)
        valid_dataset = None
        if self.data.get("validation_period", None) is not None:
            kwargs_pydataset.update({"dataset": "validation"})
            valid_dataset = self.pydataset(temporal_period = self.data["validation_period"], **kwargs_pydataset)
        ### Update metadata and save it with the new information
        self.metadata_dict = self.cont_metadata(train_dataset) 
        # self.save_metadata(self.metadata_path)
        log.info("Pydatasets ready")
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
        Performs a single forward and backward pass for a batch.

        `model` is expected to be an EDM-preconditioned backbone (e.g.
        deep4production.deep.models.diffusion.edm_precond.EDMPrecond wrapping a
        SongUNet). It accepts forward(x=r_t, sigma=sigma_t, cond_low, cond_high)
        and returns the denoised prediction D_theta. All EDM preconditioning
        (c_in, c_skip, c_out, c_noise) lives inside the preconditioner.

        Parameters
        ----------
        model : nn.Module
            Preconditioned denoiser. forward(x, sigma, cond_low, cond_high) -> D_theta.
        data : tuple
            (residual, context_low_res, context_high_res).
        optimizer : torch.optim.Optimizer
        loss_function : callable
            Signature: (target, output, sigma_t) -> scalar. The EDM λ(σ) weighting
            is applied inside the loss.
        device : str
        noise_params : dict
            Expected keys: P_mean, P_std, sigma_min, sigma_max. `sigma_data`
            is owned by the preconditioner buffer and the loss.
        is_this_training : bool
            If True, run loss.backward().

        Returns
        -------
        float : batch loss value.
        """
        # --- Noise schedule parameters ---
        P_mean = noise_params["P_mean"]
        P_std = noise_params["P_std"]
        sigma_min = noise_params["sigma_min"]
        sigma_max = noise_params["sigma_max"]

        # --- Unpack batch ---
        r, c_low, c_high = data
        batch_size = r.shape[0]
        non_blocking = (self.device_type == "cuda")

        r = r.to(device, non_blocking=non_blocking)
        if c_low is not None:
            c_low = c_low.to(device, non_blocking=non_blocking)
        if c_high is not None:
            c_high = c_high.to(device, non_blocking=non_blocking)

        # --- GPU-side normalization ---
        # Only c_low needs normalization here: r is the residual of normalized
        # values (clean target ≈ N(0, sigma_data) by construction) and c_high
        # is the regressor's prediction in normalized space — both were stored
        # to the residuals zarr already in normalized space by
        # pydataset_resdiff._forward_pass_regressor (variable names are
        # suffixed *_residual / *_normalized respectively).
        if c_low is not None and self.norm_x is not None:
            c_low = self.norm_x(c_low)

        # --- Sample noise level and corrupt the clean target — kept in fp32 ---
        sigma_t = self.sigma(P_mean, P_std, sigma_min=sigma_min, sigma_max=sigma_max, batch_size=batch_size).to(device)
        z = torch.randn_like(r)
        r_t = r + sigma_t * z

        optimizer.zero_grad(set_to_none=True)

        # --- Forward through the EDM-preconditioned model + loss under AMP autocast ---
        with self._amp_ctx():
            D_theta = model(x=r_t, sigma=sigma_t, cond_low=c_low, cond_high=c_high)
            loss = loss_function(target=r, output=D_theta, sigma_t=sigma_t)

        if is_this_training:
            if self._scaler is not None:
                self._scaler.scale(loss).backward()
            else:
                loss.backward()

        return loss.detach()
