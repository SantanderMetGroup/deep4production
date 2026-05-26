## Load libraries
import torch
from torch_geometric.data import HeteroData, Batch

## Deep4production
from deep4production.core.downscalers.downscaler import downscaler


##################################################################################################################################
class downscaler_custom(downscaler):
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

    def __init__(
        self,
        id_dir,
        input_data,
        model_file=None,
        saving_info=None,
        ensemble_size=1,
        graph=None,
        forcing_data=None,
    ):
        """
        Initializes D4P GNN4CD's downscaler.
        """
        ######### Call parent constructor to initialize common attributes #########
        super().__init__(
            id_dir=id_dir,
            input_data=input_data,
            model_file=model_file,
            saving_info=saving_info,
            ensemble_size=ensemble_size,
            graph=graph,
            forcing_data=forcing_data,
        )

    # ---------------------------------------------------------------------------------------------------------------------<
    def graphPredict(
        self, x: torch.Tensor, edge_index, model, f: torch.Tensor
    ) -> torch.Tensor:
        """
        Batched GNN prediction via PyG Batch.from_data_list.

        Builds one HeteroData per sample in the batch, concatenates them into a
        single disconnected graph so message passing operates on all samples in
        parallel, then reshapes the output back to (B, C_y, G_high).

        Parameters
        ----------
        x : (B, n_lag, C_low, G_low) if lagged, or (B, C_low, G_low) if single-step.
            Low-resolution node features (preprocessed).
        edge_index : tuple (low_to_high_edges, high_within_high_edges)
        model : GNN4CD instance
        f : (B, C_high, G_high)  high-resolution forcing features.
            Zeros tensor when no forcing data is configured.

        Returns
        -------
        torch.Tensor  (B, C_y, G_high)
        """
        B = x.shape[0]
        graphs = []
        for b in range(B):
            data_graph = HeteroData()
            data_graph["low", "to", "high"].edge_index = edge_index[0].to(self.device)
            data_graph["high", "within", "high"].edge_index = edge_index[1].to(
                self.device
            )

            # Low-res node features: (G_low, n_lag, C_low)
            if x.dim() == 4:  # (B, n_lag, C_low, G_low)
                data_graph["low"].x = x[b].permute(2, 0, 1)  # (G_low, n_lag, C_low)
            else:  # (B, C_low, G_low) — no lag
                data_graph["low"].x = x[b].T.unsqueeze(1)  # (G_low, 1, C_low)

            # High-res node features: (G_high, C_high)
            # f comes in as (B, C_high, G_high); transpose to PyG convention.
            data_graph["high"].x = f[b].T  # (G_high, C_high)

            graphs.append(data_graph)

        batched = Batch.from_data_list(graphs)

        with torch.inference_mode():
            # GNN4CD.forward returns (C_y, B * G_high) after its internal permute.
            pred = model(batched)

        # Reshape (C_y, B * G_high) → (B, C_y, G_high)
        C_y = pred.shape[0]
        return pred.reshape(C_y, B, self.G_y).permute(1, 0, 2)
