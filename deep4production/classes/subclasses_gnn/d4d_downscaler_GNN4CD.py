## Load libraries
import torch
from torch_geometric.data import HeteroData
## Deep4production
from deep4production.classes.d4d_downscaler import d4d_downscaler
##################################################################################################################################
class d4d_downscaler_custom(d4d_downscaler):
    def __init__(self, id_dir, input_data, model_file=None, saving_info=None, ensemble_size=1, graph=None, forcing_data=None):
        """
        Initializes D4D GNN4CD's downscaler.
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
    def to_graph(self, x, edge_index, f=["N/A"]):
        # --- Build HeteroData structure --- 
        data_graph = HeteroData()
        data_graph["low", "to", "high"].edge_index = edge_index[0].to(self.device)
        data_graph["high", "within", "high"].edge_index = edge_index[1].to(self.device)
        data_graph['low'].x  = torch.permute(x[0], (2,0,1))   # permute to shape: (N_low, seq, c_low)
        if f[0] == "N/A":
            data_graph['high'].x = torch.zeros(self.G_y, 1, device=self.device) # shape: (N_high, c_high)
        else: 
            data_graph['high'].x = f[0].to(self.device) # permute to shape: (N_high, c_high)
        pred = model(data_graph).unsqueeze(0).cpu().numpy()
        return pred

