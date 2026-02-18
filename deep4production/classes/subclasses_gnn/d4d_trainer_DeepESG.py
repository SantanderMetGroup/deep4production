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

        # --- Get arrays as defined in the pydataset class, and graph information ---
        x, y, f = data
        edges, coords_latent_list, pos_features_latent_list = edge_index
        

        # --- Edges ---
        for key in edges:
            edges[key] = edges[key].to(device)

        # --- Positional features ---
        pos_features_latent_list = [
                torch.tensor(pos, dtype=torch.float, device=device)
                for pos in pos_features_latent_list
        ]

        # --- Loop over samples since PyG does not accept samples as dimension ---
        target_list = []
        prediction_list = []
        num_samples = x.shape[0]
        for s in range(num_samples):
            # --- Get sample arrays ---
            xs = x[s].to(device) 
            ys = y[s].to(device)
            Cy, Gy = ys.shape
            target_list.append(ys)

            # --- Permute to match graph structure: (nodes, channels)
            xs = torch.permute(xs, (1,0))

            # --- High-res forcings --- 
            # print(f.shape)
            if f[s] == "N/A":
                fs = torch.zeros(Gy, Cy, device=device) # shape: (N_high, c_high)
            else: 
                fs = f[s].to(device)
                fs = torch.permute(fs, (1,0))
            
            # --- Feed features to the denoiser deep learning model ---
            pred = model(xs, edges_dict=edges, pos_list=pos_features_latent_list, f=fs)
            prediction_list.append(pred)
        
        # --- Concatenate sample ---
        prediction = torch.stack(prediction_list, dim=0)  # (num_samples, C, G)
        # print(f"prediction: {prediction.shape}")
        target = torch.stack(target_list, dim=0)  # (num_samples, C, G)
        # print(f"target: {target.shape}")


        # --- Compute loss ---
        optimizer.zero_grad()
        loss = loss_function(target=target, output=prediction)

        # --- Backpropagation ---
        if is_this_training:
            loss.backward()

        # --- Return ---
        return loss.item() 
