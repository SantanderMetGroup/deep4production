import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from deep4downscaling.utils.general import get_func_from_string

# ------------------------------------------------------------------------------------------
def plot_psd(
    data,
    compute_psd=True,
    compute_psd_kwargs=None,
    loglog=True,
    ax=None,
    title=None,
    labels=None,
    colors=None
):
    """
    Plot PSD for multiple xarray DataArrays in one figure.

    Parameters
    ----------
    data : list[xr.DataArray]
        List of data arrays to compute PSD on.
    compute_psd : bool, optional
        If True, compute PSD using power_spectral_density().
    compute_psd_kwargs : dict, optional
        Passed to power_spectral_density().
    loglog : bool, optional
        Use log–log axes.
    ax : matplotlib Axes, optional
        Axis to plot on.
    title : str, optional
        Title of the figure.

    Returns
    -------
    ax : matplotlib Axes
    """

    # --- Colors and labels ---
    if colors is None:
        colors = ["blue"]
    if labels is None:
        labels = ["PSD"]

    # --- Placeholder for compute_psd_kwargs ---
    if compute_psd_kwargs is None:
        compute_psd_kwargs = {}

    # --- Import PSD function ---
    psd_func = get_func_from_string(
        module_string="deep4downscaling.utils.diagnostics",
        func_string="power_spectral_density"
    )

    # --- Compute PSDs for all datasets ---
    psd_list = []
    for da in data:
        aux = psd_func(da=da, **compute_psd_kwargs)
        dim_avg = [d for d in aux.dims if d != "freq"]
        if len(dim_avg) > 0:
            aux = aux.mean(dim=dim_avg)
        psd_list.append(aux)

    # ---Create axis if needed ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    # --- Plot each PSD ---
    for i, psd in enumerate(psd_list):
        # Frequency dimension
        freq = psd["freq"].values
        # psd values
        y = psd.values
        ax.plot(freq, y, label=labels[i], color=colors[i])

    # --- Plotting details ---
    ## Axis
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ## Labels
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power Spectral Density")
    ## Title
    if title is not None:
        ax.set_title(title)
    ## Grid
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    ## Return
    fig = ax.figure
    return fig 


# ------------------------------------------------------------------------------------------
def plot_psd_spatial(data):
    return plot_psd(data=data, compute_psd=True, compute_psd_kwargs={"dim": "point"}, loglog=True, ax=None, title=None, labels=["target", "prediction"], colors=["blue", "orange"])


# ------------------------------------------------------------------------------------------
def plot_psd_temporal(data):
    return plot_psd(data=data, compute_psd=True, compute_psd_kwargs={"dim": "time"}, loglog=True, ax=None, title=None, labels=["target", "prediction"], colors=["blue", "orange"])