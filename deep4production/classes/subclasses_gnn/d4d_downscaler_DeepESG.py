## Load libraries
import torch
from torch_geometric.data import HeteroData
## Deep4production
from deep4production.classes.d4d_downscaler import d4d_downscaler
##################################################################################################################################
class d4d_downscaler_custom(d4d_downscaler):
    def __init__(self, id_dir, input_data, model_file=None, saving_info=None, ensemble_size=1, graph=None, forcing_data=None):
        """
        Initializes D4D DeepESG's downscaler.
        """
        ######### Call parent constructor to initialize common attributes #########
        super().__init__(
            id_dir=id_dir,
            input_data=input_data,
            model_file=model_file,
            saving_info=saving_info,
            ensemble_size=ensemble_size,
            graph=graph,
            forcing_data=forcing_data
        )

    # ---------------------------------------------------------------------------------------------------------------------<
    def graphPredict(self, x, edge_index, model, f="N/A"):

        # print(f"x shape: {x.shape}")
        edges, coords_latent_list, pos_features_latent_list = edge_index

        # --- Move to device ---
        for key in edges:
            edges[key] = edges[key].to(self.device)
        pos_features_latent_list = [
            torch.tensor(pos, dtype=torch.float, device=self.device)
            for pos in pos_features_latent_list
        ]
        xs = x[0].to(self.device) 

        # --- Permute to match graph structure: (nodes, channels)
        xs = torch.permute(xs, (1,0))
        # print(f"xs shape: {xs.shape}")

        # --- High-res forcings --- 
        # print(f.shape)
        if f == "N/A":
            C_y = len(self.vars_y)
            fs = torch.zeros(self.G_y, C_y, device=self.device) # shape: (N_high, c_high)
        else: 
            fs = f[0].to(self.device)
            fs = torch.permute(fs, (1,0))
        
        # --- Feed features to the denoiser deep learning model ---
        with torch.no_grad():
            pred = model(xs, edges_dict=edges, pos_list=pos_features_latent_list, f=fs).unsqueeze(0)

        # --- Return ---
        return pred.cpu().numpy()



