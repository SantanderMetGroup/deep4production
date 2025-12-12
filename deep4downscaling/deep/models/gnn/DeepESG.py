import zarr
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from deep4downscaling.utils.general import latlon_to_xyz


# -------------------------------------------------------------------------------------------------------------------
def plot_coords(coords_low, coords_high, coords_latent_1, coords_latent_2):
    fig = plt.figure(figsize=(12, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # --- Plot the grid points ---
    ax.scatter(coords_low[:,1], coords_low[:,0],
               s=10, label="LOW", alpha=0.7, color="green",
               transform=ccrs.PlateCarree())

    ax.scatter(coords_high[:,1], coords_high[:,0],
               s=4, label="HIGH", alpha=0.7, color="orange", marker="x",
               transform=ccrs.PlateCarree())

    ax.scatter(coords_latent_1[:,1], coords_latent_1[:,0],
               s=2, label="LATENT_1", alpha=0.7, color="blue",
               transform=ccrs.PlateCarree())
    
    ax.scatter(coords_latent_2[:,1], coords_latent_2[:,0],
            s=0.5, label="LATENT_2", alpha=0.7, color="red",
            transform=ccrs.PlateCarree())

    # --- Coastlines ---
    ax.coastlines(resolution="110m", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)

    # --- Formatting ---
    ax.set_title("Grid Coordinates: Low, Latent, High")
    ax.legend(loc="upper right")
    # ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4)

    # --- Save ---
    plt.savefig("DeepESG_graph.png", dpi=200, bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------------------------------------------------------------
def plot_bipartite_graph(coords_A, coords_B, edges_A_to_B,
                          title="A → B Graph",
                          savepath="A_to_B_graph.png",
                          node_size_A=12,
                          node_size_B=12,
                          edge_alpha=0.25):
    """
    Plot nodes_A, nodes_B, and edges connecting A -> B.

    Args:
        coords_A: (N_A, 2) array of (lat, lon) for source nodes
        coords_B: (N_B, 2) array of (lat, lon) for target nodes
        edges_A_to_B: (2, E) tensor where:
          edges[0] indexes coords_A
          edges[1] indexes coords_B
        title: plot title
        savepath: output filename
    """

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": proj})

    # -----------------------------------------
    # Scatter nodes
    # -----------------------------------------
    ax.scatter(coords_A[:,1], coords_A[:,0],
               s=node_size_A, color="blue", alpha=0.9,
               label="Layer A", transform=proj)

    ax.scatter(coords_B[:,1], coords_B[:,0],
               s=node_size_B, color="red", alpha=0.9,
               label="Layer B", transform=proj)

    # -----------------------------------------
    # Draw edges
    # -----------------------------------------
    src = edges_A_to_B[0].numpy()
    dst = edges_A_to_B[1].numpy()

    for s, d in zip(src, dst):
        ax.plot(
            [coords_A[s,1], coords_B[d,1]],
            [coords_A[s,0], coords_B[d,0]],
            linewidth=0.6,
            color="black",
            alpha=edge_alpha,
            transform=proj,
        )

    # -----------------------------------------
    # Formatting
    # -----------------------------------------
    ax.set_title(title, fontsize=14)
    ax.coastlines("110m", linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax.legend()
    # ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4)

    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[OK] saved {savepath}")

# -------------------------------------------------------------------------------------------------------------------
def build_coords_latent(coords_ref, degree_spacing=1.0):
    """
    Build a regular lat-lon grid of latent node coordinates.

    Args:
        coords_ref: array of shape (N, 2), columns = [lat, lon]
        degree_spacing: float, grid spacing in degrees

    Returns:
        coords_latent: array (N_latent, 2) of [lat, lon]
    """

    lats = coords_ref[:, 0]
    lons = coords_ref[:, 1]

    # --- Normalize lon range if necessary ---
    # If data spans > 300° it's probably a 0..360 grid
    if (lons.max() - lons.min()) > 300:
        # Convert to 0..360
        lons = (lons + 360) % 360

    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    # --- Create regularly spaced grid ---
    lat_grid = np.arange(lat_min + 2*degree_spacing, lat_max - degree_spacing, degree_spacing)
    lon_grid = np.arange(lon_min + 2*degree_spacing, lon_max - degree_spacing, degree_spacing)

    # Mesh
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    
    coords_latent = np.stack([LAT.ravel(), LON.ravel()], axis=1)

    return coords_latent

# -------------------------------------------------------------------------------------------------------------------
def build_hierarchical_graphs(
    data_high: str,
    data_low: str,
    k_low_to_low: int = 8,
    k_prev_to_next: int = 4,
    k_latent_to_latent: int = 8,
    k_last_to_high: int = 4,
    degree_spacing_latent: list = [2.0, 1.0],
):
    """
    Build hierarchical graph:
        low -> low
        low -> latent_0
        latent_0 -> latent_1
        ...
        latent_{n-2} -> latent_{n-1}
        latent_{n-1} -> high

    Returns:
        edges: dict of edge_index tensors
        coords_latent_list: list of lat/lon arrays
        pos_features_latent: list of arrays
    """
    import zarr
    import numpy as np
    import torch
    from sklearn.neighbors import NearestNeighbors
    from deep4downscaling.utils.general import latlon_to_xyz

    edges = {}
    coords_latent_list = []
    pos_features_latent_list = []

    # ---------------------------------------------------------
    # Load coordinates
    # ---------------------------------------------------------
    z_high = zarr.open(data_high, mode="r")
    lat_high = np.array(z_high.attrs["lats"])
    lon_high = np.array(z_high.attrs["lons"])
    coords_high_latlon = np.stack([lat_high, lon_high], axis=1)
    coords_high_xyz = latlon_to_xyz(lat=lat_high, lon=lon_high)
    N_high = coords_high_latlon.shape[0]

    z_low = zarr.open(data_low, mode="r")
    lat_low = np.array(z_low.attrs["lats"])
    lon_low = np.array(z_low.attrs["lons"])
    coords_low_latlon = np.stack([lat_low, lon_low], axis=1)
    coords_low_xyz = latlon_to_xyz(lat=lat_low, lon=lon_low)
    N_low = coords_low_latlon.shape[0]

    # ---------------------------------------------------------
    # 1. Low → Low edges
    # ---------------------------------------------------------
    nn_low = NearestNeighbors(n_neighbors=k_low_to_low).fit(coords_low_xyz)
    _, idx_low_to_low = nn_low.kneighbors(coords_low_xyz)

    low_edges = []
    for i in range(N_low):
        for j in idx_low_to_low[i]:
            low_edges.append((i, j))

    edges["low_to_low"] = torch.tensor(low_edges, dtype=torch.long).t()

    # ---------------------------------------------------------
    # 2. Hierarchical latent refinements
    # ---------------------------------------------------------
    prev_latlon = coords_low_latlon
    prev_xyz = coords_low_xyz

    for level, spacing in enumerate(degree_spacing_latent):
        print(f"Building latent level {level} with spacing {spacing}°")

        # Build finer grid
        coords_latent_latlon = build_coords_latent(
            coords_ref=prev_latlon,
            degree_spacing=spacing
        )
        coords_latent_xyz = latlon_to_xyz(
            lat=coords_latent_latlon[:, 0],
            lon=coords_latent_latlon[:, 1]
        )
        N_latent = coords_latent_xyz.shape[0]

        coords_latent_list.append(coords_latent_latlon)

        # Positional features
        pos_features = np.stack([
            np.sin(coords_latent_latlon[:, 0]),
            np.cos(coords_latent_latlon[:, 0]),
            np.sin(coords_latent_latlon[:, 1]),
            np.cos(coords_latent_latlon[:, 1]),
        ], axis=1)
        pos_features_latent_list.append(pos_features)

        # ----------------------------------------------
        # Connect prev layer → this latent layer
        # (prev is either low or latent_{level-1})
        # ----------------------------------------------
        nn_prev = NearestNeighbors(n_neighbors=k_prev_to_next).fit(prev_xyz)
        _, idx_prev_to_latent = nn_prev.kneighbors(coords_latent_xyz)

        prev_name = "low" if level == 0 else f"latent_{level-1}"

        edges_list = []
        for j_latent in range(N_latent):
            for j_prev in idx_prev_to_latent[j_latent]:
                edges_list.append((j_prev, j_latent))

        edges[f"{prev_name}_to_latent_{level}"] = torch.tensor(edges_list, dtype=torch.long).t()

        # ----------------------------------------------
        # Latent → Latent (intra-layer)
        # ----------------------------------------------
        nn_lat = NearestNeighbors(n_neighbors=k_latent_to_latent).fit(coords_latent_xyz)
        _, idx_lat_to_lat = nn_lat.kneighbors(coords_latent_xyz)

        intra_edges = []
        for j in range(N_latent):
            for k in idx_lat_to_lat[j]:
                intra_edges.append((j, k))

        edges[f"latent_{level}_to_latent_{level}"] = torch.tensor(intra_edges, dtype=torch.long).t()

        # Update for next iteration
        prev_latlon = coords_latent_latlon
        prev_xyz = coords_latent_xyz

    # ---------------------------------------------------------
    # 3. Last latent → high
    # ---------------------------------------------------------
    nn_last = NearestNeighbors(n_neighbors=k_last_to_high).fit(prev_xyz)
    _, idx_last_to_high = nn_last.kneighbors(coords_high_xyz)

    last_latent_name = f"latent_{len(degree_spacing_latent)-1}"

    edges_list = []
    for i_high in range(N_high):
        for j_latent in idx_last_to_high[i_high]:
            edges_list.append((j_latent, i_high))

    edges[f"{last_latent_name}_to_high"] = torch.tensor(edges_list, dtype=torch.long).t()

    return edges, coords_latent_list, pos_features_latent_list

# -------------------------------------------------------------------------------------------------------------------
class Block(nn.Module):
    """
    Hierarchical GNN block with:
        - M rounds of intra-level GraphConv
        - Bipartite projection prev → next
        - Positional forcing
        - Residual → BatchNorm → ReLU (GraphCast style)
    """

    def __init__(self, channels_prev, channels_next, M=3):
        super().__init__()
        self.M = M
        self.channels_next = channels_next

        # Intra-level message passing (prev → prev)
        self.convs_intra = nn.ModuleList([
            GraphConv(channels_prev, channels_prev) for _ in range(M)
        ])

        # Bipartite projection prev → next
        self.conv_proj = GraphConv((channels_prev, channels_next), channels_next, aggr='mean')

        # # Residual projection
        # self.proj_residual = nn.Linear(channels_prev, channels_next)

        # # Positional forcing
        # self.pos_proj = nn.Linear(pos_dim, channels_next)

        # BatchNorm after projection + residual
        self.norm_prev = nn.BatchNorm1d(channels_prev)

    def forward(self, x, edge_intra, edge_prev_to_next, pos_next):
        """
        Args:
            x_prev:             (N_prev, channels_prev)
            edge_intra:         edges within prev graph
            edge_prev_to_next:  edges from prev → next
            pos_next:           (N_next, pos_dim)
        Returns:
            x_next: (N_next, channels_next)
        """

        # # 0) Add positional forcing
        # x = x + self.pos_proj(pos_next)

        # 1) Intra-level message passing
        for gconv in self.convs_intra:
            x = gconv(x, edge_intra)
            x = self.norm_prev(x)
            x = F.relu(x)

        # 2) Bipartite projection prev → next
        N_next = pos_next.size(0)
        x_target_placeholder = torch.zeros(N_next, self.channels_next, device=x.device)
        x_proj = self.conv_proj((x, x_target_placeholder), edge_prev_to_next)

        return x_proj


# -------------------------------------------------------------------------------------------------------------------
class HighResForcingPostprocessing(nn.Module):
    """
    High-res forcing postprocessing block:
    2 Conv1x1 (Linear) layers with intermediate latent_dim,
    each followed by BatchNorm and ReLU.
    """
    def __init__(self, high_res_forcing_dim, latent_dim, num_output_vars):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(high_res_forcing_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(latent_dim, num_output_vars),
            nn.BatchNorm1d(num_output_vars),
            nn.ReLU()
        )

    def forward(self, x):
        """
        x: [N_high, high_res_forcing_dim]
        returns: [N_high, num_output_vars]
        """
        h = self.block1(x)
        out = self.block2(h)
        return out


# -------------------------------------------------------------------------------------------------------------------
class DeepESG(nn.Module):
    """
    Hierarchical GNN:
        low-res -> latent_0 -> ... -> latent_{n-1} -> high-res
    """

    def __init__(self, 
                 channels,
                 high_res_forcing_input_channels=0,
                 high_res_forcing_latent_channels=0,
                 message_passing_passes=3,
                 loss_function_name=None,
                 output_activation=None):
        """
        Args:
            channels_input: int, input features on low-res graph
            channels_hidden: int, hidden feature dimension in all latent graphs
            pos_dims: list[int], positional feature dimension for each latent graph
            M: int, number of intra-level GraphConv passes
        """
        super().__init__()

        ## --- Determine output features based on loss function ---
        self.loss_function_name = loss_function_name
        if self.loss_function_name == "NLLGaussianLoss":
            channels[-1] = channels[-1] * 2
        if self.loss_function_name == "NLLBerGammaLoss":
            channels[-1] = 1 * 3
        self.num_output_vars = channels[-1] 

        # --- Init blocks ---
        self.blocks = nn.ModuleList()
        num_blocks = len(channels) - 1
        for i in range(num_blocks):
            block = Block(channels[i], channels[i+1], M=message_passing_passes)
            self.blocks.append(block)


        # --- High-res forcing postprocessing ---
        if high_res_forcing_input_channels > 0:
            self.forcings_mapping = HighResForcingPostprocessing(
                high_res_forcing_dim=high_res_forcing_input_channels,
                latent_dim=high_res_forcing_latent_channels,
                num_output_vars=channels[-1]
            )
        else:
            self.forcings_mapping = None

        # --- Optional per-variable activation ---
        self.output_activation = nn.ModuleDict()
        self._activation_map = {i: nn.Identity() for i in range(self.num_output_vars)}
        if output_activation:
            for var_name, spec in output_activation.items():
                idx = spec["idx"]
                act_class = spec.get("module", nn)
                act = getattr(act_class, spec["name"], nn.Identity)()
                self.output_activation[var_name] = act
                self._activation_map[idx] = act

    def forward(self, x_low, edges_dict, pos_list, f):
        """
        Args:
            x_low: (N_low, channels_input) features of low-res graph
            edges_dict: dict of edge_index tensors for all connections, keys:
                - "low_to_low"
                - "low_to_latent_0"
                - "latent_i_to_latent_i" (intra-level)
                - "latent_i_to_latent_{i+1}" (projection)
                - "latent_{n-1}_to_high"
            pos_list: list of positional feature tensors for each latent/high graph
        Returns:
            x_high: (N_high, channels_hidden)
        """
        x_prev = x_low

        for i, block in enumerate(self.blocks):
            # intra-level edges
            if i == 0:
                edge_intra = edges_dict["low_to_low"]
                edge_prev_to_next = edges_dict["low_to_latent_0"]
                pos_next = pos_list[i]
            elif i == len(self.blocks)-1:
                edge_intra = edges_dict[f"latent_{i-1}_to_latent_{i-1}"]
                edge_prev_to_next = edges_dict[f"latent_{i-1}_to_high"]
                pos_next = f
            else:
                edge_intra = edges_dict[f"latent_{i-1}_to_latent_{i-1}"]
                edge_prev_to_next = edges_dict[f"latent_{i-1}_to_latent_{i}"]
                pos_next = pos_list[i]

            # Block forward
            x_next = block(x_prev, edge_intra, edge_prev_to_next, pos_next)

            # Update for next level
            if i != len(self.blocks)-1:
                x_prev = x_next

        # Final output projection
        out = x_next
        if (self.forcings_mapping is not None) and (f is not None):
            out = out + self.forcings_mapping(f)

        # --- Apply per-variable activation ---
        if self.loss_function_name == "NLLBerGammaLoss":
            p = torch.sigmoid(out[:,0,None])
            log_shape = torch.sigmoid(out[:,1,None])
            log_scale = torch.sigmoid(out[:,2,None])
            out = torch.cat((p, log_shape, log_scale), dim = 1)
        else:
            activated = []
            for idx in range(self.num_output_vars):
                act = self._activation_map.get(idx, nn.Identity())
                activated.append(act(out[:, idx:idx+1]))
            out = torch.cat(activated, dim=1)

        # --- Permute ---
        out = out.permute(1, 0)  # permute to shape: (channels_high, N_high)

        # --- Return ---
        return out



# ----------------------------------------------------------------------------------------------------------------
# ## RUN EXAMPLE
# data_low = "/gpfs/projects/meteo/WORK/banoj/projects/CORDEX-BENCH/data/zarrs/files/UPSRCM_1961-1980.zarr"
# data_high  = "/gpfs/projects/meteo/WORK/banoj/projects/CORDEX-BENCH/data/zarrs/files/RCM_1961-1980.zarr"
# k_low_to_low = 8
# k_prev_to_next = 8
# k_latent_to_latent = 8
# k_last_to_high = 8
# degree_spacing_latent = [1, 0.5]

# z_high = zarr.open(data_high, mode="r")
# lat_high = np.array(z_high.attrs["lats"])
# lon_high = np.array(z_high.attrs["lons"])
# N_high = len(lat_high)
# coords_high_latlon = np.stack([lat_high, lon_high], axis=1)

# z_low = zarr.open(data_low, mode="r")
# lat_low = np.array(z_low.attrs["lats"])
# lon_low = np.array(z_low.attrs["lons"])
# N_low = len(lat_low)
# coords_low_latlon = np.stack([lat_low, lon_low], axis=1)

# graph_info = build_hierarchical_graphs(
#     data_high = data_high,
#     data_low = data_low,
#     k_low_to_low = k_low_to_low,
#     k_prev_to_next = k_prev_to_next,
#     k_latent_to_latent = k_latent_to_latent,
#     k_last_to_high = k_last_to_high,
#     degree_spacing_latent = degree_spacing_latent
# )

# plot_coords(coords_low_latlon, coords_high_latlon, graph_info[1][0], graph_info[1][1])
# plot_bipartite_graph(coords_A=coords_low_latlon, 
#                      coords_B=graph_info[1][0], 
#                      edges_A_to_B=graph_info[0]["low_to_latent_0"],
#                      title="Low → Latent_0 Graph",
#                      savepath="Low_to_Latent0_graph.png",
#                      node_size_A=10,
#                      node_size_B=4,
#                      edge_alpha=0.25)
# plot_bipartite_graph(coords_A=graph_info[1][-1], 
#                      coords_B=coords_high_latlon, 
#                      edges_A_to_B=graph_info[0]["latent_1_to_high"],
#                      title="Latent_N → High Graph",
#                      savepath="LatentN_to_High_graph.png",
#                      node_size_A=10,
#                      node_size_B=4,
#                      edge_alpha=0.25)

# gnn = HierarchicalGNN(channels = [25,10,10,1],
#                       high_res_forcing_input_channels=0,
#                       high_res_forcing_latent_channels=0,
#                       message_passing_passes=3)

