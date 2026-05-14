import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from deep4production.utils.zarr import open_zarr_store


def build_graph(
    data_high: str,
    data_low: str,
    nearest_neighbours_high_to_high: int = 8,
    nearest_neighbours_low_to_high: int = 9,
):
    """
    Builds a hetero PyG graph matching the structure required by the GNN4CD model
    (Blasone et al., Environmental Data Science 2024).

    Paper specification:
      - Each High node is linked to its k=9 nearest Low nodes.
      - High–High edges use 8 nearest distinct neighbours (self excluded) and
        are bidirectional (for each (i, j) pair we emit both (i, j) and (j, i)).

    Node types:
        - low
        - high
    Edge types:
        - ('low', 'to', 'high')
        - ('high', 'within', 'high')

    Parameters
    ----------
    data_high : str
        Path to high-resolution Zarr file (reads lats/lons from attrs).
    data_low : str
        Path to low-resolution Zarr file.
    nearest_neighbours_high_to_high : int
        Number of *distinct* neighbours per high node (paper: 8; self excluded).
    nearest_neighbours_low_to_high : int
        Number of low nodes each high node is linked to (paper: 9).

    Returns
    -------
    (low_to_high_edges, high_edges) : torch.LongTensor
        Each of shape (2, num_edges), PyG convention.
    """
    # ------------------------------------------------------------
    # 1. Load coordinates from Zarr metadata
    # ------------------------------------------------------------
    z_high = open_zarr_store(data_high, fmt="auto")
    lat_high = np.array(z_high["latitudes"][:])
    lon_high = np.array(z_high["longitudes"][:])
    N_high = len(lat_high)
    coords_high = np.stack([lat_high, lon_high], axis=1)

    z_low = open_zarr_store(data_low, fmt="auto")
    lat_low = np.array(z_low["latitudes"][:])
    lon_low = np.array(z_low["longitudes"][:])
    coords_low = np.stack([lat_low, lon_low], axis=1)

    # ------------------------------------------------------------
    # 2. HIGH → HIGH edges (k distinct neighbours, bidirectional)
    # ------------------------------------------------------------
    # Ask for k+1 neighbours and drop the self-neighbour (distance 0).
    k_hh = nearest_neighbours_high_to_high
    nn_high = NearestNeighbors(n_neighbors=k_hh + 1).fit(coords_high)
    _, idx_hh = nn_high.kneighbors(coords_high)
    idx_hh = idx_hh[:, 1:]  # drop self column

    # Build directed (i, j) pairs from kNN, then symmetrize for bidirectionality.
    src = np.repeat(np.arange(N_high), k_hh)
    dst = idx_hh.reshape(-1)
    pairs = np.stack(
        [
            np.concatenate([src, dst]),
            np.concatenate([dst, src]),
        ],
        axis=0,
    )
    # Deduplicate (i, j) == (j, i) overlaps from symmetrisation.
    pairs = np.unique(pairs, axis=1)
    high_edges = torch.from_numpy(pairs).to(torch.long).contiguous()

    # ------------------------------------------------------------
    # 3. LOW → HIGH edges (k nearest lows per high node; directed)
    # ------------------------------------------------------------
    k_lh = nearest_neighbours_low_to_high
    nn_low = NearestNeighbors(n_neighbors=k_lh).fit(coords_low)
    _, idx_lh = nn_low.kneighbors(coords_high)  # (N_high, k_lh)

    low_src = idx_lh.reshape(-1)
    high_dst = np.repeat(np.arange(N_high), k_lh)
    low_to_high_edges = (
        torch.from_numpy(np.stack([low_src, high_dst], axis=0))
        .to(torch.long)
        .contiguous()
    )

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

    def __init__(
        self,
        c_low,
        c_rnn_out,
        pred_dim=1,
        c_high=None,
        channels_downscaler_low_in=128,
        num_lagged_predictors=1,
        num_layers_rnn=2,
        channels_downscaler_out=64,
        channels_downscaler_base=64,
    ):
        super(GNN4CD, self).__init__()

        num_lagged_predictors += 1  # include current time step → sequence length

        # ── Pre-processor: GRU over (N_low, seq_len, c_low) → (N_low, seq_len, c_rnn_out)
        self.rnn = nn.GRU(c_low, c_rnn_out, num_layers_rnn, batch_first=True)

        # ── Dense: projects the flattened GRU output to the low-node encoding.
        # Input dim = c_rnn_out * seq_len (GRU output flattened).
        # NOTE: the upstream reference code uses `h_in * seq_l` here, which is
        # technically wrong (should be h_hid * seq_l) but works because the
        # default h_in == h_hid. We use the correct c_rnn_out so the model
        # remains valid when c_low != c_rnn_out.
        self.dense = nn.Sequential(
            nn.Linear(c_rnn_out * num_lagged_predictors, channels_downscaler_low_in),
            nn.ReLU(),
        )

        # ── Downscaler: low → high GraphConv.
        # `c_high` is the number of static features on each high node (e.g. DEM,
        # land-use, day-of-year). When no high-node features are available, fall
        # back to 1 dummy channel so GraphConv's (src_ch, dst_ch) pair is valid.
        self.c_high = c_high if c_high else 1
        self.downscaler = geometric_nn.Sequential(
            "x, edge_index",
            [
                (
                    GraphConv(
                        (channels_downscaler_low_in, self.c_high),
                        out_channels=channels_downscaler_out,
                        aggr="mean",
                    ),
                    "x, edge_index -> x",
                )
            ],
        )

        self.processor = geometric_nn.Sequential(
            "x, edge_index",
            [
                (geometric_nn.BatchNorm(channels_downscaler_out), "x -> x"),
                (
                    GATv2Conv(
                        in_channels=channels_downscaler_out,
                        out_channels=channels_downscaler_base,
                        heads=2,
                        dropout=0.2,
                        aggr="mean",
                        add_self_loops=True,
                        bias=True,
                    ),
                    "x, edge_index -> x",
                ),
                (geometric_nn.BatchNorm(channels_downscaler_base * 2), "x -> x"),
                nn.ReLU(),
                (
                    GATv2Conv(
                        in_channels=channels_downscaler_base * 2,
                        out_channels=channels_downscaler_base,
                        heads=2,
                        dropout=0.2,
                        aggr="mean",
                        add_self_loops=True,
                        bias=True,
                    ),
                    "x, edge_index -> x",
                ),
                (geometric_nn.BatchNorm(channels_downscaler_base * 2), "x -> x"),
                nn.ReLU(),
                (
                    GATv2Conv(
                        in_channels=channels_downscaler_base * 2,
                        out_channels=channels_downscaler_base,
                        heads=2,
                        dropout=0.2,
                        aggr="mean",
                        add_self_loops=True,
                        bias=True,
                    ),
                    "x, edge_index -> x",
                ),
                (geometric_nn.BatchNorm(channels_downscaler_base * 2), "x -> x"),
                nn.ReLU(),
                (
                    GATv2Conv(
                        in_channels=channels_downscaler_base * 2,
                        out_channels=channels_downscaler_base,
                        heads=2,
                        dropout=0.2,
                        aggr="mean",
                        add_self_loops=True,
                        bias=True,
                    ),
                    "x, edge_index -> x",
                ),
                (geometric_nn.BatchNorm(channels_downscaler_base * 2), "x -> x"),
                nn.ReLU(),
                (
                    GATv2Conv(
                        in_channels=channels_downscaler_base * 2,
                        out_channels=channels_downscaler_base,
                        heads=1,
                        dropout=0.0,
                        aggr="mean",
                        add_self_loops=True,
                        bias=True,
                    ),
                    "x, edge_index -> x",
                ),
                nn.ReLU(),
            ],
        )

        self.predictor = nn.Sequential(
            nn.Linear(channels_downscaler_base, channels_downscaler_base),
            nn.ReLU(),
            nn.Linear(channels_downscaler_base, 32),
            nn.ReLU(),
            nn.Linear(32, pred_dim),
        )

    def forward(self, data):
        """
        Forward pass for GNN4CD model.
        Parameters:
            data (dict): Dictionary containing 'low' and 'high' node data and edge indices.
        Returns:
            torch.Tensor: Output predictions for high-resolution nodes.
        """
        x_low = data["low"].x  # shape: (N_low, seq_len, c_low)
        x_high = data["high"].x
        encod_rnn, _ = self.rnn(x_low)  # shape (N_low, seq_len, h_hid)
        encod_rnn = encod_rnn.flatten(start_dim=1)  # becomes (N_low, seq_len * h_hid)
        encod_rnn = self.dense(encod_rnn)
        encod_low2high = self.downscaler(
            (encod_rnn, x_high), data["low", "to", "high"].edge_index
        )
        encod_high = self.processor(
            encod_low2high, data.edge_index_dict[("high", "within", "high")]
        )
        x_high = self.predictor(encod_high)
        # Permute to (pred_dim, N_high) so the trainer can unsqueeze(0) to get
        # (B=1, C=pred_dim, G=N_high), matching deep4production's (B, C, G)
        # loss convention. Upstream returns (N_high, pred_dim) instead.
        x_high = x_high.permute(1, 0)
        return x_high
