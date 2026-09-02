"""
Pooling of several independent simulations into one training set.

d4p's ``predictors.paths`` / ``predictands.paths`` lists concatenate along TIME:
``utils.temporal.get_sample_map`` keys every sample by calendar date and takes
the first store that has it. That is the right behaviour for a run split over
several files (historical + scenario, one file per decade), but it makes pooling
impossible: two simulations of the SAME period resolve every date to the first
store, so the second contributes nothing at all — silently, with no error.

This module pools instead. One ``pydataset`` is built per source and the samples
are concatenated, so N simulations covering the same years give N times the
samples. Because each source is a self-contained dataset, its predictors and
predictands are paired within that source by construction rather than by index
alignment across two parallel path lists.

Each source also normalizes with its OWN statistics, on the CPU
(``pydataset(normalize_on_cpu=True)``), which is what lets simulations with
different mean states be pooled at all: they arrive at the model in a common
standardized space. A single GPU-side normalizer, which is what the trainer uses
for a single-source run, cannot express that.

Authors:
    Jorge Baño-Medina
"""

from torch.utils.data import ConcatDataset

from deep4production.utils.log import get_logger

log = get_logger("pydataset.multi")


# Attributes that must agree across sources for the pooled samples to be
# collatable into one batch and for the metadata taken from sources[0] to
# describe all of them.
_MUST_MATCH = (
    "vars_x",
    "vars_y",
    "vars_f",
    "H_x",
    "W_x",
    "H_y",
    "W_y",
    "G_x",
    "G_y",
    "transform_to_2D_x",
    "transform_to_2D_y",
    "num_lagged_x",
    "num_lagged_y",
)


# -------------------------------------------------------------------------
def merge_source_block(shared_block, source_block):
    """
    Combine a shared ``predictors:`` / ``predictands:`` / ``forcings:`` block
    with one source's overrides into the dict ``pydataset`` expects.

    The recipe splits these deliberately: everything that MUST be identical
    across sources for the pool to be collatable (``variables``, ``operator``,
    ``transform_to_2D``, ``num_lagged``, and the normalizer's ``default`` plus
    its per-variable methods) is written once in the shared block, while what is
    necessarily per-source (``paths`` and the normalizer's ``path_reference``)
    lives in the source entry. Splitting it this way means a source cannot
    accidentally be given a different channel order.

    Merging is shallow, except ``normalizer`` which is merged one level deeper
    so a source contributes ``path_reference`` without dropping the shared
    ``default`` and per-variable method overrides.
    """
    merged = dict(shared_block or {})
    source_block = dict(source_block or {})
    source_norm = source_block.pop("normalizer", None)
    merged.update(source_block)
    if source_norm is not None:
        merged["normalizer"] = {**(merged.get("normalizer") or {}), **source_norm}
    return merged


########################################################################################################
########################################################################################################
class MultiSourceDataset(ConcatDataset):
    """
    Concatenation of one pydataset per source, presenting the pydataset metadata
    API the trainer expects.

    Parameters
    ----------
    datasets : list
        One already-constructed pydataset per source, in recipe order.
    names : list of str, optional
        Source names, used for logging only. Defaults to ``source_0``, ...
    """

    def __init__(self, datasets, names=None):
        if not datasets:
            raise ValueError("❌ MultiSourceDataset requires at least one source.")
        self.names = list(names) if names else [f"source_{i}" for i in range(len(datasets))]
        if len(self.names) != len(datasets):
            raise ValueError(
                f"❌ Got {len(datasets)} sources but {len(self.names)} names."
            )
        self._check_consistency(datasets, self.names)
        super().__init__(datasets)

        # The per-source split is the one thing that is invisible in the loss
        # curve and expensive to get wrong, so it is logged up front.
        log.info(
            "Pooled %d sources, %d samples total: %s",
            len(datasets),
            len(self),
            ", ".join(
                f"{n}={len(d)}" for n, d in zip(self.names, datasets)
            ),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _check_consistency(datasets, names):
        """
        Refuse to pool sources that disagree on grid, channels or lags.

        This is a hard error, not a warning: metadata (and therefore the model's
        input shape, the output coordinates and the checkpoint) is taken from
        the first source, so a mismatch elsewhere would either crash deep in the
        collate/forward with an opaque shape error, or — for a variable-order
        mismatch, where the shapes still line up — train happily on scrambled
        channels.
        """
        ref = datasets[0]
        for name, ds in zip(names[1:], datasets[1:]):
            for attr in _MUST_MATCH:
                got, expected = getattr(ds, attr, None), getattr(ref, attr, None)
                if got != expected:
                    raise ValueError(
                        f"❌ Source '{name}' disagrees with '{names[0]}' on "
                        f"{attr}: {got!r} vs {expected!r}. Every pooled source "
                        "must share the same grid, variables (in the same "
                        "order) and lag configuration."
                    )

    # -------------------------------------------------------------------------
    # Metadata API — delegated to the first source, which _check_consistency has
    # guaranteed is representative of all of them.
    # -------------------------------------------------------------------------
    @property
    def reference(self):
        """The source whose metadata describes the pool."""
        return self.datasets[0]

    def get_vars(self):
        return self.reference.get_vars()

    def get_lagged_info(self):
        return self.reference.get_lagged_info()

    def get_coords(self):
        return self.reference.get_coords()

    def get_transform2D(self):
        return self.reference.get_transform2D()

    def get_spatial_dims(self):
        return self.reference.get_spatial_dims()

    def get_num_gridpoints(self):
        return self.reference.get_num_gridpoints()

    def get_operator_info(self, predictands=False):
        return self.reference.get_operator_info(predictands=predictands)

    def get_forcings_info(self):
        return self.reference.get_forcings_info()

    def get_target_samples(self):
        """
        Validation-period ground truth for monitoring diagnostics.

        Only the reference source's targets: the monitoring predictions come
        from a downscaler pointed at a single predictor store
        (``trainer._setup_monitor_inputs``), so pooling the targets here would
        compare against simulations the prediction never saw.
        """
        return self.reference.get_target_samples()

    @staticmethod
    def _resolve_normalizer_info(*args, **kwargs):
        """Static passthrough — the trainer calls this off the dataset object."""
        from deep4production.core.pydatasets.pydataset import pydataset

        return pydataset._resolve_normalizer_info(*args, **kwargs)
