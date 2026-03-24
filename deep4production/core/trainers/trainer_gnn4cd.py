## Load libraries
import os
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_dense_batch
## Deep4production
from deep4production.core.trainers.trainer import trainer
##################################################################################################################################
class trainer_custom(trainer):
    """
    Custom trainer class for GNN4CD models using PyTorch Geometric.
    Purpose: Handles graph-based batch training, builds HeteroData structures, and computes loss for GNN models.
    Parameters:
        data (dict): Dataset configuration.
        dataloader (dict): Dataloader parameters.
        id_dir (str): Experiment directory.
        model_info (dict): Model, loss, saving, and training parameters.
        graph (dict): Graph configuration for GNN models.
        d4dpy (dict): Custom pydataset configuration.
        Mlflow (dict): MLflow tracking configuration.
        edge_index_path (str): Path to the pre-computed edge index file for the static graph.
    """

    def __init__(self, data, dataloader, id_dir, model_info, graph, d4dpy, Mlflow, edge_index_path):
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
            Mlflow=Mlflow
        )

        # ---- Build ONE static graph ----
        print("⚙️ Building static graph for GNN4CD...")
        edge_index = torch.load(edge_index_path)

        self.graph = HeteroData()
        self.graph["low", "to", "high"].edge_index = edge_index[0]
        self.graph["high", "within", "high"].edge_index = edge_index[1]

        self.graph_on_device = False

    # -------------------------------------------------------------------------
    def model_backprop(self, model, data, optimizer, loss_function, device, is_this_training=True):
        """
        Performs a single forward and backward pass for a batch of size 1 using graph-based input.
        Purpose: Places features in HeteroData structure, feeds to GNN model, computes loss, and performs backpropagation.
        Parameters:
            model: PyTorch model.
            data: Tuple of input, target, and forcing arrays.
            optimizer: PyTorch optimizer.
            loss_function: Loss function callable.
            device: Device string ('cpu' or 'cuda').
            is_this_training (bool): Whether to perform backpropagation.
        Returns:
            float: Loss value for the batch.
        """

        x, y, f = data
        if f[0] == "N/A":
            f = torch.zeros_like(y)

        if not self.graph_on_device:
            self.graph = self.graph.to(device)
            self.graph_on_device = True

        # ---- Reshape inputs ----
        x = x[0].permute(2, 0, 1).to(device)   # from (sample, seq, C, G_low) → (G_low, seq, C)
        y = y[0].permute(1, 0).to(device)  # from (sample, C, G_low) → (G_high, C)
        f = f[0].permute(1, 0).to(device)  # from (sample, C, G_low) → (G_high, C)

        # ---- Attach features to static graph ----
        self.graph["low"].x = x
        self.graph["high"].x = f
        self.graph["high"].y = y

        # ---- Forward ----
        prediction = model(self.graph)  # (N_high, C)

        # --- Compute loss ---
        y = y.permute(1,0).unsqueeze(0) # permute to (num_vars, num_high_nodes) and add time dimension for loss computation
        prediction = prediction.unsqueeze(0) # permute to (num_vars, num_high_nodes) and add time dimension for loss computation
        optimizer.zero_grad()
        loss = loss_function(target=y, output=prediction)

        # --- Backpropagation ---
        if is_this_training:
            loss.backward()

        # --- Return ---
        return loss.item()








