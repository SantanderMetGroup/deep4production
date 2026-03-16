import zarr
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def build_graph(
    data_high: str,
    data_low: str,
    nearest_neighbours_high_to_high: int = 8,
    nearest_neighbours_low_to_high: int = 4
):
    """
    Builds a hetero PyG graph matching the structure required by the GNN4CD model.
    Node types:
        - low
        - high
    Edge types:
        - ('low', 'to', 'high')
        - ('high', 'within', 'high')
    Parameters:
        data_high (str): Path to high-resolution Zarr file.
        data_low (str): Path to low-resolution Zarr file.
        nearest_neighbours_high_to_high (int): Number of neighbors for high-to-high edges.
        nearest_neighbours_low_to_high (int): Number of neighbors for low-to-high edges.
    Returns:
        tuple: (low_to_high_edges, high_edges) as torch tensors.
    """

    # ------------------------------------------------------------
    # 1. Load coordinates from Zarr metadata
    # ------------------------------------------------------------
    z_high = zarr.open(data_high, mode="r")
    lat_high = np.array(z_high.attrs["lats"])
    lon_high = np.array(z_high.attrs["lons"])
    N_high = len(lat_high)
    coords_high = np.stack([lat_high, lon_high], axis=1)

    z_low = zarr.open(data_low, mode="r")
    lat_low = np.array(z_low.attrs["lats"])
    lon_low = np.array(z_low.attrs["lons"])
    N_low = len(lat_low)
    coords_low = np.stack([lat_low, lon_low], axis=1)

    # ------------------------------------------------------------
    # 2. Build HIGH → HIGH graph (KNN)
    # ------------------------------------------------------------
    nn_high = NearestNeighbors(n_neighbors=nearest_neighbours_high_to_high).fit(coords_high)
    _, idx_hh = nn_high.kneighbors(coords_high)

    high_edges = []
    for i in range(N_high):
        for j in idx_hh[i]:
            high_edges.append((i, j))  # keep directed
    high_edges = torch.tensor(high_edges, dtype=torch.long).t().contiguous()

    # ------------------------------------------------------------
    # 3. Build LOW → HIGH edges
    # ------------------------------------------------------------
    nn_low = NearestNeighbors(n_neighbors=nearest_neighbours_low_to_high).fit(coords_low)
    _, idx_lh = nn_low.kneighbors(coords_high)  # each high → nearest low nodes

    low_to_high_edges = []
    for i_high in range(N_high):
        for j in idx_lh[i_high]:
            low_to_high_edges.append((j, i_high))

    low_to_high_edges = torch.tensor(low_to_high_edges, dtype=torch.long).t().contiguous()

    # Return
    return low_to_high_edges, high_edges



###########################################################
# Original code at: https://github.com/valebl/GNN4CD/blob/main/models/GNN4CD_model.py
import torch.nn as nn
import torch_geometric.nn as geometric_nn
from torch_geometric.nn import GATv2Conv, GraphConv

class GNN4CD(nn.Module):
    """
    Graph Neural Network for Climate Downscaling (GNN4CD).
    Purpose: Processes low and high resolution climate data using graph convolutions and attention layers.
    Parameters:
        c_low (int): Input channels for low-resolution nodes.
        c_rnn_out (int): Output channels for RNN encoder.
        pred_dim (int): Output prediction dimension.
        c_high (int): Input channels for high-resolution nodes.
        channels_downscaler_low_in (int): Channels for low-resolution downscaler input.
        num_lagged_predictors (int): Number of lagged predictors.
        num_layers_rnn (int): Number of RNN layers.
        channels_downscaler_out (int): Channels for downscaler output.
        channels_downscaler_base (int): Base channels for downscaler.
    """
    
    def __init__(self, c_low, c_rnn_out, pred_dim=1, c_high=0, channels_downscaler_low_in=128, num_lagged_predictors=1, num_layers_rnn=2, channels_downscaler_out=64, channels_downscaler_base=64):
        super(GNN4CD, self).__init__()

        num_lagged_predictors += 1 # include current time step

        # input shape (N,L,Hin)
        self.rnn = nn.Sequential(
            nn.GRU(c_low, c_rnn_out, num_layers_rnn, batch_first=True),
        )

        self.dense = nn.Sequential(
            nn.Linear(c_low*num_lagged_predictors, channels_downscaler_low_in),
            nn.ReLU()
        )

        self.c_high = c_high if c_high is not None else 1  # placeholder
        self.downscaler = geometric_nn.Sequential('x, edge_index', [
            (GraphConv(
                (channels_downscaler_low_in, self.c_high), 
                out_channels=channels_downscaler_out, 
                aggr='mean'
            ), 'x, edge_index -> x')
        ])
        
        self.processor = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(channels_downscaler_out), 'x -> x'),
            (GATv2Conv(in_channels=channels_downscaler_out, out_channels=channels_downscaler_base, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(channels_downscaler_base*2), 'x -> x'), 
            nn.ReLU(),
            (GATv2Conv(in_channels=channels_downscaler_base*2, out_channels=channels_downscaler_base, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(channels_downscaler_base*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=channels_downscaler_base*2, out_channels=channels_downscaler_base, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(channels_downscaler_base*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=channels_downscaler_base*2, out_channels=channels_downscaler_base, heads=2, dropout=0.2, aggr='mean', add_self_loops=True, bias=True),'x, edge_index -> x'),
            (geometric_nn.BatchNorm(channels_downscaler_base*2), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(in_channels=channels_downscaler_base*2, out_channels=channels_downscaler_base, heads=1, dropout=0.0, aggr='mean', add_self_loops=True, bias=True), 'x, edge_index -> x'),
            nn.ReLU(),
            ])
    
        self.predictor = nn.Sequential(
            nn.Linear(channels_downscaler_base, channels_downscaler_base),
            nn.ReLU(),
            nn.Linear(channels_downscaler_base, 32),
            nn.ReLU(),
            nn.Linear(32, pred_dim)
            )

    def forward(self, data):
        """
        Forward pass for GNN4CD model.
        Parameters:
            data (dict): Dictionary containing 'low' and 'high' node data and edge indices.
        Returns:
            torch.Tensor: Output predictions for high-resolution nodes.
        """
        x_low = data['low'].x       # shape: (N_low, seq_len, c_low)
        x_high = data['high'].x 
        encod_rnn, _ = self.rnn(x_low) # shape (N_low, seq_len, h_hid)
        encod_rnn = encod_rnn.flatten(start_dim=1) # becomes (N_low, seq_len * h_hid)
        encod_rnn = self.dense(encod_rnn)
        encod_low2high  = self.downscaler((encod_rnn, x_high), data["low", "to", "high"].edge_index)
        encod_high = self.processor(encod_low2high , data.edge_index_dict[('high','within','high')])
        x_high = self.predictor(encod_high)
        x_high = x_high.permute(1, 0)  # permute to shape: (pred_dim, N_high)
        return x_high