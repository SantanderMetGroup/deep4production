"""
Training monitors — a single abstraction over the experiment-tracking backends.

The trainer used to carry two parallel, deeply-nested ``if self.Mlflow is not
None`` / ``if self.tracker is not None`` branches scattered across the training
loop. This module collapses them behind one small interface so the loop just
calls ``self.monitor.<hook>(...)`` and stays readable. Two backends implement
the interface (they are mutually exclusive, enforced in ``cli/train.py``):

  * :class:`MLflowMonitor`  — logs scalars/figures/checkpoints to an MLflow
    server (wraps :mod:`deep4production.utils.mlflow`).
  * :class:`TrackerMonitor` — writes figures + CSV to ``<id_dir>/outputs/tracker/``
    on the local filesystem (wraps :mod:`deep4production.utils.tracker`).

The base :class:`Monitor` is an all-no-op null object, used on non-main ranks
and when no backend is configured, so the trainer never needs a None check.

Design note: monitors are pure logging *sinks*. They never touch the model or
the downscaler — the trainer owns prediction and passes results in. Diagnostic
hooks receive a ``predict`` callback so the (potentially expensive) validation
prediction only runs on the epochs a backend actually logs.
"""

import os

from deep4production.utils.log import get_logger
from deep4production.utils.tracker import tracker_write_losses, tracker_epoch_logs
from deep4production.utils.paths import tracker_dir

# NB: ``mlflow`` and ``deep4production.utils.mlflow`` are imported lazily inside
# ``MLflowMonitor`` so the dependency-free d4p-tracker backend (and the no-op
# base monitor) can run without MLflow installed at all.

log = get_logger("monitor")


# =========================================================================
class Monitor:
    """No-op base monitor (null object). Every hook is a safe no-op."""

    backend = "none"
    # The trainer sets up the validation-prediction machinery only when a
    # backend actually needs predictions for its diagnostics.
    needs_predictions = False
    # True when any configured metric is computed in the model's normalized
    # [-1,1] space, so the trainer transforms the fields before logging.
    needs_model_space = False
    # Preserves the historical coupling whereby step-checkpointing was gated on
    # an active MLflow run (see the trainer's "save every n steps" block).
    logs_checkpoints = False

    def log_losses(self, epoch, train_losses, valid_losses):
        """Record the epoch's train/val losses."""

    def should_save_checkpoint(self, epoch):
        """Return True on epochs where a backend wants a checkpoint logged."""
        return False

    def log_checkpoint(self, path):
        """Log a just-saved checkpoint file as a backend artifact."""

    def maybe_log_diagnostics(self, epoch, vars, tgt, predict, to_model_space=None):
        """
        Run + log validation diagnostics if this epoch is due. ``predict`` is a
        zero-arg callback returning the prediction Dataset; it is only invoked
        when the backend decides to log (so prediction stays lazy).
        ``to_model_space`` is an optional callable mapping a physical-units
        Dataset to the model's normalized [-1,1] space (used when a metric
        declares ``space: model``).
        """

    def on_training_end(self, epoch_best, vars, tgt, best_checkpoint_path, predict_from_best):
        """Final hook: log the best checkpoint / on-best figures."""

    def close(self):
        """Tear down the backend (e.g. end the MLflow run)."""


# =========================================================================
class _CadenceMixin:
    """Shared 'every N epochs' gate with an internal reference epoch."""

    @staticmethod
    def _due(epoch, ref, every):
        """True when ``every`` is set and at least that many epochs have passed."""
        return every is not None and (epoch - ref) >= every


# =========================================================================
class MLflowMonitor(Monitor, _CadenceMixin):
    """Logs to an MLflow tracking server (run already started by the CLI)."""

    backend = "mlflow"
    logs_checkpoints = True

    def __init__(self, cfg):
        import mlflow

        self.cfg = cfg
        # Tags (the run is live by now — started in cli/train.py).
        for key, value in cfg.get("tags", {}).items():
            if value is not None:
                mlflow.set_tag(key, value)
        self.diagnostics = cfg.get("diagnostics", None)
        self.compute_every = cfg.get("compute_diagnostics_every_n_epochs", None)
        self.save_ckpt_every = cfg.get("save_checkpoint_every_n_epochs", None)
        # Predictions are needed for periodic diagnostics and for on-best figures.
        self.needs_predictions = self.diagnostics is not None
        self._diag_ref = 0
        self._ckpt_ref = 0

    def log_losses(self, epoch, train_losses, valid_losses):
        import mlflow

        mlflow.log_metric("train_loss_epoch", float(train_losses[-1]), step=int(epoch))
        if valid_losses:
            mlflow.log_metric("val_loss_epoch", float(valid_losses[-1]), step=int(epoch))

    def should_save_checkpoint(self, epoch):
        if self._due(epoch, self._ckpt_ref, self.save_ckpt_every):
            self._ckpt_ref = epoch
            return True
        return False

    def log_checkpoint(self, path):
        import mlflow

        mlflow.log_artifact(path, artifact_path="checkpoints")

    def maybe_log_diagnostics(self, epoch, vars, tgt, predict, to_model_space=None):
        if self.diagnostics is None:
            return
        if not self._due(epoch, self._diag_ref, self.compute_every):
            return
        from deep4production.utils.mlflow import (
            mlflow_scalars_logs,
            mlflow_figures_logs,
        )

        self._diag_ref = epoch
        prd = predict()
        scalars = self.diagnostics.get("scalars", None)
        if scalars is not None:
            mlflow_scalars_logs(tgt, prd, vars, scalars, epoch)
        figures = self.diagnostics.get("figures", None)
        if figures is not None and not figures.get("on_best", False):
            mlflow_figures_logs(tgt, prd, vars, figures, epoch)
        if self.diagnostics.get("xai_scalars", None) is not None:
            log.warning("XAI scalars logs not implemented; skipping.")

    def on_training_end(self, epoch_best, vars, tgt, best_checkpoint_path, predict_from_best):
        import mlflow

        if self.cfg.get("save_best", False) and best_checkpoint_path is not None:
            mlflow.log_artifact(best_checkpoint_path, artifact_path="checkpoints")
        figures = (self.diagnostics or {}).get("figures", None)
        if figures is not None and figures.get("on_best", False):
            from deep4production.utils.mlflow import mlflow_figures_logs

            prd = predict_from_best()
            mlflow_figures_logs(tgt, prd, vars, figures, epoch_best)

    def close(self):
        import mlflow

        mlflow.end_run()


# =========================================================================
class TrackerMonitor(Monitor, _CadenceMixin):
    """Writes figures + CSV to ``<id_dir>/outputs/tracker/`` (no external server)."""

    backend = "tracker"

    def __init__(self, cfg, id_dir):
        self.tracker_dir = tracker_dir(id_dir)
        os.makedirs(self.tracker_dir, exist_ok=True)
        self.metrics = cfg.get("metrics", None)
        self.maps = cfg.get("maps", None)
        self.compute_every = cfg.get("compute_diagnostics_every_n_epochs", None)
        self.needs_predictions = self.metrics is not None or self.maps is not None
        self.needs_model_space = _any_model_space(self.metrics)
        self._diag_ref = 0
        # Cached so the per-epoch snapshot can redraw the loss curve.
        self._train_losses = []
        self._valid_losses = None

    def log_losses(self, epoch, train_losses, valid_losses):
        self._train_losses = train_losses
        self._valid_losses = valid_losses
        # Cheap CSV, refreshed every epoch; the figure renders with the snapshot.
        tracker_write_losses(train_losses, valid_losses, self.tracker_dir)

    def maybe_log_diagnostics(self, epoch, vars, tgt, predict, to_model_space=None):
        if not self.needs_predictions:
            return
        if not self._due(epoch, self._diag_ref, self.compute_every):
            return
        self._diag_ref = epoch
        prd = predict()
        # Transform to normalized [-1,1] model space only when a metric needs it.
        tgt_model = prd_model = None
        if self.needs_model_space and to_model_space is not None:
            tgt_model = to_model_space(tgt)
            prd_model = to_model_space(prd)
        tracker_epoch_logs(
            tgt=tgt,
            prd=prd,
            vars=vars,
            tracker_dir=self.tracker_dir,
            epoch=epoch,
            train_losses=self._train_losses,
            valid_losses=self._valid_losses,
            metrics_info=self.metrics,
            maps_info=self.maps,
            tgt_model=tgt_model,
            prd_model=prd_model,
        )


# =========================================================================
def _any_model_space(metrics):
    """True if any metric entry in the (default + per-variable) config asks for
    ``space: model`` — only the dict form can."""
    if not metrics:
        return False
    for entries in metrics.values():
        for entry in entries:
            if isinstance(entry, dict) and entry.get("space") == "model":
                return True
    return False


# =========================================================================
def build_monitor(mlflow_cfg, tracker_cfg, *, id_dir, is_main):
    """
    Construct the active monitor for this rank.

    Non-main ranks always get the no-op :class:`Monitor` so monitoring I/O
    happens on rank 0 only. The two backends are mutually exclusive (also
    enforced in ``cli/train.py``; re-checked here as a guard).
    """
    if not is_main:
        return Monitor()
    if mlflow_cfg is not None and tracker_cfg is not None:
        raise ValueError(
            "'Mlflow' and 'tracker' are mutually exclusive; configure only one."
        )
    if mlflow_cfg is not None:
        return MLflowMonitor(mlflow_cfg)
    if tracker_cfg is not None:
        return TrackerMonitor(tracker_cfg, id_dir)
    return Monitor()
