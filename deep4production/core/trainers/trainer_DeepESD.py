## Load libraries
import os
import numpy as np
import torch
from torch_geometric.data import HeteroData
## Deep4production
from deep4production.classes.d4d_trainer import d4d_trainer
##################################################################################################################################
class d4d_trainer_custom(d4d_trainer):
    """
    Custom trainer class for DeepESD models using ensemble members.
    Purpose: Handles batch training, ensemble prediction, and loss computation for CNN-based models.
    Parameters:
        data (dict): Dataset configuration.
        dataloader (dict): Dataloader parameters.
        id_dir (str): Experiment directory.
        model_info (dict): Model, loss, saving, and training parameters.
        graph (dict): Graph configuration (optional).
        d4dpy (dict): Custom pydataset configuration.
        Mlflow (dict): MLflow tracking configuration.
    """
    def __init__(self, data, dataloader, id_dir, model_info, graph, d4dpy, Mlflow):

        ######### Call parent constructor to initialize common attributes #########
        super().__init__(
            data=data,
            dataloader=dataloader,
            id_dir=id_dir,
            model_info=model_info,
            graph=graph,
            d4dpy=d4dpy,
            Mlflow=Mlflow
        )

    # -------------------------------------------------------------------------
    def model_backprop(self, model, data, optimizer, loss_function, device, is_this_training=True, members=2):
        """
        Performs a single forward and backward pass for a batch using ensemble prediction.
        Purpose: Runs forward passes for each ensemble member, stacks predictions, computes loss, and performs backpropagation.
        Parameters:
            model: PyTorch model.
            data: Tuple of input, target, and forcing arrays.
            optimizer: PyTorch optimizer.
            loss_function: Loss function callable.
            device: Device string ('cpu' or 'cuda').
            is_this_training (bool): Whether to perform backpropagation.
            members (int): Number of ensemble members.
        Returns:
            float: Loss value for the batch.
        """
        # --- Get arrays ---
        x, y, f = data
        x = x.to(device)
        y = y.to(device)

        # --- Forcing ---
        if f[0] != "N/A":
            f = f.to(device)
        else:
            B, Cy, *spatial = y.shape
            f = torch.zeros(B, Cy, *spatial, device=device)

        # --- Forward pass for each ensemble member ---
        prediction_list = []
        for m in range(members):
            pred_m = model(x, f)  # shape: (B, C, H, W) or (B, C, G)
            prediction_list.append(pred_m)

        # Stack along new ensemble dimension -> (B, M, C, H, W) or (B, M, C, G)
        prediction = torch.stack(prediction_list, dim=1)
        # print(f"prediction shape: {prediction.shape}")

        # --- Compute loss ---
        optimizer.zero_grad()
        loss = loss_function(target=y, output=prediction)

        # --- Backpropagation ---
        if is_this_training:
            loss.backward()
            optimizer.step()

        return loss.item()
