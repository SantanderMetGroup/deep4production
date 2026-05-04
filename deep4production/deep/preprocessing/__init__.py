"""
Model-side preprocessors for deep4production.

Modules
-------
normalizer : InputNormalizer
    Vectorized affine normalizer registered as a model preprocessor;
    replaces per-sample CPU normalization in pydataset with a single
    fused in-place op on the GPU tensor.
"""

from deep4production.deep.preprocessing.normalizer import InputNormalizer

__all__ = ["InputNormalizer"]
