import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from deep4production.utils.general import get_func_from_string

# ------------------------------------------------------------------------------------------
def plot_psd(
    data,
    compute_psd_func,
    compute_psd_kwargs={},
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


    # --- Import PSD function ---
    psd_func = get_func_from_string(
        module_string="deep4production.utils.diagnostics",
        func_string=compute_psd_func
    )

    # --- Compute PSDs for all datasets ---
    psd_list = []
    for da in data:
        aux = psd_func(da=da, **compute_psd_kwargs)
        dim_avg = [d for d in aux.dims if d not in ["freq", "wavenumber"]]
        if len(dim_avg) > 0:
            aux = aux.mean(dim=dim_avg)
        psd_list.append(aux)

    # ---Create axis if needed ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    # --- Plot each PSD ---
    for i, psd in enumerate(psd_list):
        # Frequency dimension
        if "freq" in psd.dims:
            x_vals = psd["freq"].values
            x_label = "Frequency"
        elif "wavenumber" in psd.dims:
            x_vals = psd["wavenumber"].values
            x_label = "Wavenumber"
        # psd values
        y = psd.values
        ax.plot(x_vals, y, label=labels[i], color=colors[i])

    # --- Plotting details ---
    ## Axis
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ## Labels
    ax.set_xlabel(x_label)
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
def plot_psd_spatial(data, reshape_spatial_dims):
    return plot_psd(data=data, compute_psd_func="radially_averaged_power_spectral_density", compute_psd_kwargs={"reshape_spatial_dims": reshape_spatial_dims}, 
        loglog=True, ax=None, title=None, labels=["target", "prediction"], colors=["blue", "orange"])


# ------------------------------------------------------------------------------------------
def plot_psd_temporal(data):
    return plot_psd(data=data, compute_psd_func="power_spectral_density", 
        loglog=True, ax=None, title=None, labels=["target", "prediction"], colors=["blue", "orange"])