import math
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

def plot_date_from_1D_spatial_field(
    data,
    set_extent,
    central_longitude=0,
    date=None,
    time_index=None,
    vmin=None,
    vmax=None,
    cmap="YlGnBu",
    titles=None,
    suptitle="",
    figsize=(10, 8),
    cbar_label="Value",
    diff=False,
    vminDiff=None,
    vmaxDiff=None,
    cmapDiff="RdBu_r",
):
    """
    Plot multiple 1D-spatial fields (lat/lon points) on maps, optionally with a difference panel.
    Each element in `data` is an xarray.DataArray with dims ('time', 'point') and coords 'lat','lon'.
    """

    # --- Setup figure layout ---
    proj = ccrs.PlateCarree()
    n = len(data)
    ncols = min(n, 3)
    nrows = math.ceil(n / 3)

    # Add 1 extra column if showing diff panel
    extra_cols = 1 if diff else 0

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols + extra_cols,
        figsize=figsize,
        subplot_kw={"projection": proj},
        squeeze=False,
    )
    axes = axes.flatten()

    # --- Choose a single time slice for all fields ---
    if date is not None:
        target_time = np.datetime64(date)
    elif time_index is not None:
        target_time = None
    else:
        target_time = None

    def select_time(da):
        """Return da(time=selected) if da has a time coordinate"""
        nonlocal target_time

        if "time" not in da.coords:
            return da, None

        if target_time is not None:  # date provided
            da_sel = da.sel(time=target_time)
            t_str = str(da_sel.time.values)[:10]

        elif time_index is not None:  # index provided
            da_sel = da.isel(time=time_index)
            t_str = str(da.time.values[time_index])[:10]

        else:  # default = time=0
            da_sel = da.isel(time=0)
            t_str = str(da.time.values[0])[:10]

        return da_sel, t_str

    # Use suptitle = date unless user provided
    main_title = suptitle

    # --- Loop over each DataArray ---
    selected_fields = []  # lat, lon, values
    for i, da in enumerate(data):
        ax = axes[i]

        da_sel, t_str = select_time(da)
        if t_str is not None and main_title == "":
            main_title = t_str  # auto-set only if user didn't specify

        lat = da_sel["lat"].values
        lon = da_sel["lon"].values
        vals = da_sel.values
        selected_fields.append(vals)

        sc = ax.scatter(
            lon,
            lat,
            c=vals,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=12,
            transform=ccrs.PlateCarree(),
        )

        ax.coastlines(resolution="10m")
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.set_title(titles[i] if titles is not None else f"Field {i+1}")

        # Mean annotation
        ax.text(
            0.02,
            0.95,
            f"Mean: {np.nanmean(vals):.2f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="black", boxstyle="round"),
        )

    # --- Shared colorbar ---
    cbar = fig.colorbar(
        axes[0].collections[0],
        ax=axes[:n],
        orientation="horizontal",
        fraction=0.03,
        pad=0.04,
    )
    cbar.set_label(cbar_label)

    # ---------------------------------------------------
    # --- Difference panel (Field2 - Field1) ------------
    # ---------------------------------------------------
    if diff:
        if n < 2:
            raise ValueError("diff=True but fewer than 2 fields were provided")

        diff_ax = axes[n]  # next available axis

        # Compute difference
        vals_ref = selected_fields[0]
        vals_diff = selected_fields[1] - vals_ref

        # Use vmin/vmax for diff
        if vminDiff is None:
            vminDiff = np.nanpercentile(vals_diff, 2)
        if vmaxDiff is None:
            vmaxDiff = np.nanpercentile(vals_diff, 98)

        # Plot diff scatter
        lat = da_sel["lat"].values
        lon = da_sel["lon"].values

        sc2 = diff_ax.scatter(
            lon,
            lat,
            c=vals_diff,
            cmap=cmapDiff,
            vmin=vminDiff,
            vmax=vmaxDiff,
            s=12,
            transform=ccrs.PlateCarree(),
        )

        diff_ax.coastlines(resolution="10m")
        diff_ax.add_feature(cfeature.BORDERS, linestyle=":")
        diff_ax.set_title("Difference: Field 2 - Field 1")

        diff_ax.text(
            0.02,
            0.95,
            f"Mean: {np.nanmean(vals_diff):.2f}",
            transform=diff_ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="black", boxstyle="round"),
        )

        # Colorbar for diff
        cbar2 = fig.colorbar(
            sc2, ax=diff_ax, fraction=0.03, pad=0.04, orientation="horizontal"
        )
        cbar2.set_label("Difference")

    # --- Suptitle ---
    fig.suptitle(main_title, fontsize=15)

    # --- Return ---
    return fig