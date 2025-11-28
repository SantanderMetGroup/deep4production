import math
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

# ------------------------------------------------------------------------------------------
def plot_date_from_1D_spatial_field(
    data,
    set_extent,
    central_longitude=0,
    date=None,              
    time_index=None,        
    vmin=None,              
    vmax=None,
    cmap='YlGnBu',
    titles=None,
    suptitle="",
    figsize=(8, 8),
    cbar_label="Value"
):
    """
    Plot multiple DataArrays on the same map.

    Parameters
    ----------
    data : list[xr.DataArray]
        Each DA has dims ('time', 'point') with 'lat' and 'lon' coords.
    date : np.datetime64 or str, optional
        Select a specific date to plot.
    time_index : int, optional
        Alternative to `date`.
    vmin, vmax : float, optional
        Colorbar range.
    cmap : str, optional
        Colormap.
    fig_title : str, optional
        Title for the figure.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    fig : matplotlib Figure
    """

    # --- Figure + GeoAxes ---
    proj = ccrs.PlateCarree()
    n = len(data)
    ncols = min(n, 3)
    nrows = math.ceil(n / 3)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        subplot_kw={'projection': proj},
        squeeze=False
    )
    axes = axes.flatten()

    # -------------------------
    # --- Loop over panels ----
    for i, da in enumerate(data):
        ax = axes[i]
        # --- Select time slice? ---
        if "time" in da.coords:
            if date is not None:
                datenp = np.datetime64(date)
                da_sel = da.sel(time=datenp)
            elif time_index is not None:
                da_sel = da.isel(time=time_index)
                date = str(da.time.values[time_index])[:10]
            else:
                da_sel = da.isel(time=0)  # default
                date = str(da.time.values[0])[:10]
            suptitle = date
        else:
            da_sel = da

        # --- Plot ---
        # --- Prepare coordinates ---
        lat = da_sel["lat"].values
        lon = da_sel["lon"].values
        values = da_sel.values
        # ----------------------------
        # Plot using pcolormesh-like scatter. (Since data are irregular points, not grid)
        sc = ax.scatter(
            lon, lat,
            c=values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=12,
            transform=ccrs.PlateCarree(),
        )

        # --- Add geographic features ---
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        # --- Add mean value in top-left corner ---
        mean_val = np.nanmean(values)
        ax.text(
            0.02, 0.95,
            f"Mean: {mean_val:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round')
        )

        # --- Title --- 
        if titles is not None:
            ax.set_title(titles[i])

    # --- Create a shared colorbar ---
    cbar = fig.colorbar(
        axes[0].collections[0],  # or whichever axis produced the mappable
        ax=axes,                 # link it to all subplots
        orientation='horizontal',
        fraction=0.02,
        pad=0.02
    )
    cbar.set_label(cbar_label)  # customize label

    # --- Common title for all the figure ---
    fig.suptitle(suptitle, fontsize=14)

    # --- Return ---
    return fig
