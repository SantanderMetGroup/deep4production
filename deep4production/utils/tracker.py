"""
d4p-tracker: in-house, dependency-free training monitor.

A lightweight alternative to the MLflow integration (see
``deep4production/utils/mlflow.py``). Instead of logging to an MLflow server,
d4p-tracker writes monitoring artifacts (figures + CSV) straight to
``<output_dir>/<run_ID>/tracker/`` on the local filesystem.

The two backends are mutually exclusive — exactly zero or one may be configured
in the recipe (enforced in ``cli/train.py``). This module mirrors the
``default`` + per-variable config convention and the ``get_func_from_string``
diagnostic-resolution pattern used by ``mlflow.py`` so both feel the same.

Output layout
-------------
Persistent CSV state lives at the ``tracker/`` root (appended over training);
every diagnostic epoch gets a self-contained snapshot folder holding *all* of
that epoch's figures (no extra nesting — the filename carries the detail)::

    tracker/
        losses.csv                       # one row per epoch (data, continuous)
        metric_<name>_<var>.csv          # min/mean/max history per metric
        epoch_0000/
            loss_curve.png
            evolution_<name>_<var>.png   # per-gridpoint min/mean/max so far
            map_<var>_<date>.png         # ground truth vs prediction
        epoch_0010/
            ...

Only three scalars per epoch per metric are persisted (min/mean/max) — never
the gridpoint fields — so the on-disk footprint stays bounded; each logged
epoch is drawn as a vertical min->max line with a marker at the spatial mean.
"""

import os
import csv

import numpy as np

# Headless-safe backend: training typically runs on HPC nodes with no display.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deep4production.utils.general import get_func_from_string
from deep4production.utils.log import get_logger

log = get_logger("tracker")

DIAGNOSTIC_MODULE = "deep4production.utils.diagnostics"


# -------------------------------------------------------------------------
def _epoch_dir(tracker_dir, epoch):
    """Create and return the per-epoch snapshot folder ``tracker/epoch_XXXX``."""
    path = os.path.join(tracker_dir, f"epoch_{epoch:04d}")
    os.makedirs(path, exist_ok=True)
    return path


# -------------------------------------------------------------------------
def _parse_metric_entry(entry):
    """
    Translate one configured metric entry into
    ``(display_name, diagnostic, index, take_abs)``.

    Accepted forms (same ``default`` + per-variable convention as MLflow):
      * ``"rmse"``                       -> diagnostic only
      * ``["Mean", "bias"]``             -> [index, diagnostic]
      * ``{diagnostic: bias, index: Mean, abs: true, name: ...}``  (dict form,
        the only form that exposes the ``abs`` toggle and a name override)

    ``abs`` takes the absolute value of the per-gridpoint field *before* the
    spatial min/mean/max reduction (e.g. mean |bias| instead of mean signed
    bias, which would let positive and negative errors cancel).
    """
    name = None
    take_abs = False
    if isinstance(entry, dict):
        diagnostic = entry["diagnostic"]
        index = entry.get("index")
        take_abs = bool(entry.get("abs", False))
        name = entry.get("name")
    elif isinstance(entry, (list, tuple)) and len(entry) == 2:
        index, diagnostic = entry
    else:
        diagnostic, index = str(entry), None

    if name is None:
        name = f"{index}_{diagnostic}" if index is not None else str(diagnostic)
        if take_abs:
            name = f"abs_{name}"
    return name, str(diagnostic), index, take_abs


# -------------------------------------------------------------------------
def _resolve_metric_entries(metrics_info, var):
    """Collect the metric entries applying to ``var`` (default + per-variable)."""
    entries = []
    if "default" in metrics_info:
        entries.extend(metrics_info["default"])
    if var in metrics_info:
        entries.extend(metrics_info[var])
    return entries


# -------------------------------------------------------------------------
def tracker_write_losses(train_losses, valid_losses, tracker_dir):
    """
    Persist per-epoch train/val losses to ``tracker/losses.csv`` (full rewrite,
    one row per epoch). Cheap, so it is refreshed every epoch — the matching
    figure is only rendered on diagnostic epochs (see ``tracker_epoch_logs``).
    """
    os.makedirs(tracker_dir, exist_ok=True)
    have_val = valid_losses is not None and len(valid_losses) > 0
    with open(os.path.join(tracker_dir, "losses.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for epoch, tr in enumerate(train_losses):
            val = valid_losses[epoch] if have_val and epoch < len(valid_losses) else ""
            writer.writerow([epoch, tr, val])


# -------------------------------------------------------------------------
def _plot_losses(train_losses, valid_losses, fig_dir):
    """Render the loss curve (train + optional val) into the snapshot folder."""
    have_val = valid_losses is not None and len(valid_losses) > 0
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(len(train_losses)), train_losses, label="train", color="tab:blue")
    if have_val:
        ax.plot(
            np.arange(len(valid_losses)),
            valid_losses,
            label="validation",
            color="tab:orange",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "loss_curve.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------------
def _append_metric_row(csv_path, epoch, vmin, vmean, vmax):
    """Append one (epoch, min, mean, max) row, writing the header once."""
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["epoch", "min", "mean", "max"])
        writer.writerow([epoch, vmin, vmean, vmax])


# -------------------------------------------------------------------------
def _read_metric_csv(csv_path):
    """Read back the full (epoch, min, mean, max) history for plotting."""
    epochs, mins, means, maxs = [], [], [], []
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            mins.append(float(row["min"]))
            means.append(float(row["mean"]))
            maxs.append(float(row["max"]))
    return np.array(epochs), np.array(mins), np.array(means), np.array(maxs)


# -------------------------------------------------------------------------
def _plot_metric_evolution(csv_path, png_path, title):
    """
    Draw the metric history: one vertical line per logged epoch spanning
    min->max across gridpoints, with a marker at the spatial mean.
    """
    epochs, mins, means, maxs = _read_metric_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.vlines(
        epochs,
        mins,
        maxs,
        color="tab:blue",
        alpha=0.6,
        linewidth=1.5,
        label="min–max (gridpoints)",
    )
    ax.plot(
        epochs,
        means,
        "o-",
        color="tab:red",
        markersize=4,
        linewidth=1.0,
        label="mean (gridpoints)",
    )
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel(title)
    ax.set_title(f"{title} — per-gridpoint evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------------
def _metric_logs(tgt, prd, vars, metrics_info, tracker_dir, fig_dir, epoch):
    """
    For each variable/diagnostic, compute the per-gridpoint field, reduce it to
    (min, mean, max) across gridpoints, append those to the persistent per-metric
    CSV at the tracker root, and (re)draw the evolution figure into ``fig_dir``.
    """
    for var in vars:
        logged = []
        for entry in _resolve_metric_entries(metrics_info, var):
            name, diagnostic, index, take_abs = _parse_metric_entry(entry)
            kwargs = {"target": tgt[var], "prediction": prd[var], "spatial": True}
            if index is not None:
                kwargs["index"] = index
            field = get_func_from_string(DIAGNOSTIC_MODULE, diagnostic, kwargs=kwargs)
            values = np.asarray(getattr(field, "values", field), dtype=float)
            if take_abs:
                values = np.abs(values)
            # Drop non-finite gridpoints (e.g. relbias -> inf where target ~ 0)
            # so a single bad point doesn't blow up the min/max markers.
            values = np.where(np.isfinite(values), values, np.nan)
            if values.size == 0 or np.all(np.isnan(values)):
                log.warning("[%s] metric %s is empty/all-NaN; skipping.", var, name)
                continue

            csv_path = os.path.join(tracker_dir, f"metric_{name}_{var}.csv")
            png_path = os.path.join(fig_dir, f"evolution_{name}_{var}.png")
            _append_metric_row(
                csv_path,
                epoch,
                float(np.nanmin(values)),
                float(np.nanmean(values)),
                float(np.nanmax(values)),
            )
            _plot_metric_evolution(csv_path, png_path, f"{name} [{var}]")
            logged.append(name)
        if logged:
            log.info("[%s] Tracker metrics logged: %s", var, logged)


# -------------------------------------------------------------------------
def _map_logs(tgt, prd, vars, maps_info, fig_dir, epoch):
    """
    Plot ground-truth vs. prediction maps for a few random validation dates.

    The dates are drawn with a fixed seed so the *same* dates are reused every
    epoch, making snapshots directly comparable across training.
    """
    from deep4production.viz.spatial import plot_date_from_1D_spatial_field

    num_dates = maps_info.get("num_dates", 3)
    seed = maps_info.get("seed", 42)
    diff = maps_info.get("diff", True)
    cmap = maps_info.get("cmap", "YlGnBu")
    set_extent = maps_info.get("set_extent", False)

    for var in vars:
        tgt_da, prd_da = tgt[var], prd[var]
        times = np.asarray(tgt_da["time"].values)
        n = min(num_dates, len(times))
        rng = np.random.default_rng(seed)
        dates = times[np.sort(rng.choice(len(times), size=n, replace=False))]
        for date in dates:
            date_str = str(date)[:10]
            fig = plot_date_from_1D_spatial_field(
                data=[tgt_da, prd_da],
                set_extent=set_extent,
                date=date,
                cmap=cmap,
                titles=["Ground truth", "Prediction"],
                suptitle=f"{var} — {date_str} (epoch {epoch})",
                diff=diff,
            )
            fig.savefig(
                os.path.join(fig_dir, f"map_{var}_{date_str}.png"),
                bbox_inches="tight",
                dpi=150,
            )
            plt.close(fig)
        log.info("[%s] Tracker maps logged: %d dates", var, n)


# -------------------------------------------------------------------------
def tracker_epoch_logs(
    tgt,
    prd,
    vars,
    tracker_dir,
    epoch,
    train_losses,
    valid_losses=None,
    metrics_info=None,
    maps_info=None,
):
    """
    Render a full per-epoch snapshot into ``tracker/epoch_XXXX/``: the loss
    curve, the per-gridpoint metric-evolution figures, and the random-date
    ground-truth vs. prediction maps. Persistent metric CSVs are appended at
    the tracker root. Called on diagnostic epochs only.
    """
    fig_dir = _epoch_dir(tracker_dir, epoch)
    _plot_losses(train_losses, valid_losses, fig_dir)
    if metrics_info is not None:
        _metric_logs(tgt, prd, vars, metrics_info, tracker_dir, fig_dir, epoch)
    if maps_info is not None:
        _map_logs(tgt, prd, vars, maps_info, fig_dir, epoch)
