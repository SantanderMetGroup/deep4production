## Load libraries
import os
import numpy as np
import torch
from torch_geometric.data import HeteroData
## Deep4production
from deep4production.classes.d4d_trainer import d4d_trainer
##################################################################################################################################
class d4d_trainer_custom(d4d_trainer):
    def __init__(self, data, dataloader, id_dir, model_info, graph, d4dpy, Mlflow):
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

    # -------------------------------------------------------------------------
    def model_backprop(self, model, data, optimizer, loss_function, device, edge_index, is_this_training=True):

        # --- Get arrays as defined in the pydataset class. ---
        x, y, f = data
        y = y.to(device)

        # --- Build HeteroData structure --- 
        data_graph = HeteroData()
        data_graph["low", "to", "high"].edge_index = edge_index[0].to(device)
        data_graph["high", "within", "high"].edge_index = edge_index[1].to(device)
        data_graph['low'].x  = torch.permute(x[0].to(device), (2,0,1))   # permute to shape: (N_low, seq, c_low)
        if f[0] == "N/A":
            data_graph['high'].x = torch.zeros(y.shape[2], 1, device = device) # shape: (N_high, c_high)
        else: 
            data_graph['high'].x = f[0].to(device) # permute to shape: (N_high, c_high)
        
        # --- Feed features to the denoiser deep learning model ---
        prediction = model(data_graph)

        # --- Compute loss ---
        optimizer.zero_grad()
        loss = loss_function(target=y, output=prediction.unsqueeze(0))

        # --- Backpropagation ---
        if is_this_training:
            loss.backward()

        # --- Return ---
        return loss.item() 
