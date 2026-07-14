"""Canonical on-disk layout for a d4p run directory.

Every run lives in a single self-contained directory ``id_dir = output_dir/run_ID``
that holds the run's recipes and launch scripts (``train.yaml``, ``inference.yaml``,
``train.sh``, ``inference.sh``). Everything the run *generates* goes under an
``outputs/`` subdirectory of ``id_dir``:

    id_dir/
      train.yaml  inference.yaml  train.sh  inference.sh
      outputs/
        models/        # checkpoints (.pt, metadata embedded)
        aux_files/     # caches (residuals, graphs, ...)
        predictions/   # downscaled NetCDF output
        tracker/       # d4p-tracker figures + CSV
        xai/           # d4p-explain input-attribution maps

Keeping the convention in one module means a future change to the layout is a
single edit here rather than a scattered set of string literals.
"""

import os


def outputs_dir(id_dir):
    """Directory holding everything a run generates (``id_dir/outputs``)."""
    return os.path.join(id_dir, "outputs")


def models_dir(id_dir):
    """Checkpoint directory (``id_dir/outputs/models``)."""
    return os.path.join(outputs_dir(id_dir), "models")


def aux_dir(id_dir):
    """Auxiliary-cache directory (``id_dir/outputs/aux_files``)."""
    return os.path.join(outputs_dir(id_dir), "aux_files")


def predictions_dir(id_dir):
    """Prediction-output directory (``id_dir/outputs/predictions``)."""
    return os.path.join(outputs_dir(id_dir), "predictions")


def tracker_dir(id_dir):
    """d4p-tracker output directory (``id_dir/outputs/tracker``)."""
    return os.path.join(outputs_dir(id_dir), "tracker")


def xai_dir(id_dir):
    """d4p-explain attribution-map directory (``id_dir/outputs/xai``)."""
    return os.path.join(outputs_dir(id_dir), "xai")


def resolve_id_dir(output_dir, run_ID):
    """Absolute run directory from ``output_dir`` and ``run_ID``."""
    return os.path.abspath(os.path.join(output_dir, run_ID))
