"""
This module contains utility functions for the deep learning models.

Authors: Jose González-Abad, Jorge Baño-Medina
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
import math
from deep4production.utils.general import get_func_from_string

# --------------------------------------------------------------------------------------------------------------
class StandardDataset(Dataset):

    """
    Standard Pytorch dataset for pairs of x and y. The input data must be a
    np.ndarray.

    Parameters
    ----------
    x : np.ndarray
        Array representing the predictor data

    y : np.ndarray
        Array representing the predictand data
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        x = self.x[idx, :]
        y = self.y[idx, :]
        return x, y

# --------------------------------------------------------------------------------------------------------------
def precipitation_NLL_trans(data: xr.Dataset, threshold: float) -> xr.Dataset:
    
    """
    This function performs the transformation required for training
    a model with the NLL of a Bernoulli and gamma distributions. The
    main idea is to set a threshold that defines the wet days, so the
    DL model learns the gamma only on wet days, avoiding biased amounts
    if including the amount for dry days.

    Parameters
    ----------
    data : xr.Dataset
        Data to apply the transformation to.

    threshold : float
        Threshold defining the amount for wet days.

    Returns
    -------
    xr.Dataset
        The transformed data
    """
    data_final = data.copy(deep=True)
    epsilon = 1e-06
    threshold = threshold - epsilon # Include in the distribution of wet days the threshold value
    data_final = data_final - threshold
    data_final = xr.where(cond=data_final<0, x=0, y=data_final)
    return data_final



# --------------------------------------------------------------------------------------------------------------
class EMA:
    def __init__(self, model, device, decay=0.5):
        self.model = model
        self.decay = decay
        # Initialize shadow weights as a copy of model parameters
        self.shadow = {name: param.clone().detach().to(device) for name, param in model.named_parameters() if param.requires_grad}
    def update(self):
        # Update EMA after each optimizer step
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.shadow:
                    self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param
    def apply_shadow(self):
        # Copy EMA weights to the model (for evaluation or sampling)
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

# --------------------------------------------------------------------------------------------------------------
def save_model(model, path, optimizer, epoch, global_step, train_losses, valid_losses, metadata = None, scheduler=None):
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'train_losses': train_losses[-100:],
        'valid_losses': valid_losses[-100:] if valid_losses else None,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metadata': metadata
    }
    torch.save(checkpoint, path)


# --------------------------------------------------------------------------------------------------------------
def resume_model(model, path, optimizer=None, scheduler=None, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"🚀 Model resumed from epoch {checkpoint['epoch']} step {checkpoint['global_step']}")
    return checkpoint


# --------------------------------------------------------------------------------------------------------------  
def load_model(path, map_location=None, return_metadata=False):
    # Load checkpoint
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    # Use metadata to rebuild model
    model_name, model_module, model_kwargs = checkpoint["metadata"]["model_params"]["name"], checkpoint["metadata"]["model_params"]["module"], checkpoint["metadata"]["model_params"]["kwargs"]
    model = get_func_from_string(model_module, model_name, model_kwargs)
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    # Evaluation mode
    model.eval()
    if return_metadata:
        return model, checkpoint["metadata"]
    else:
        return model
