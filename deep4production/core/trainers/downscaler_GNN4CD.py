## Load libraries
import torch
from torch_geometric.data import HeteroData
## Deep4production
from deep4production.classes.d4d_downscaler import d4d_downscaler
##################################################################################################################################
class d4d_downscaler_custom(d4d_downscaler):
    """
    Custom downscaler class for GNN4CD models using PyTorch Geometric.
    Purpose: Builds HeteroData structures for graph-based inference and generates predictions using GNN models.
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
        """
        Builds HeteroData structure for graph-based inference and generates predictions.
        Purpose: Prepares graph input, feeds to GNN model, and returns prediction array.
        Parameters:
            x (torch.Tensor): Input tensor.
            edge_index: Tuple of edge indices for graph structure.
            f (list, optional): Forcing tensor.
        Returns:
            np.ndarray: Prediction array.
        """
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

