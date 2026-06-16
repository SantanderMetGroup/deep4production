"""
Loss functions for deep4production, organized by family.

Submodules
----------
standard       : MaeLoss, MseLoss, QuantisedMSELoss, WeightedMseLoss
nll            : NLLGaussianLoss, NLLBerGammaLoss
asym           : Asym
crps           : CRPSSpectralLoss
diffusion      : WeightedDenoisingScoreMatchingLoss
classification : BinaryCrossEntropyLoss, BernoulliFocalLoss

All classes are re-exported here so that ``module: deep4production.deep.loss``
in YAML recipes continues to resolve every loss by name without change.
"""

from deep4production.deep.loss.standard import (
    MaeLoss,
    MseLoss,
    QuantisedMSELoss,
    WeightedMseLoss,
)
from deep4production.deep.loss.nll import NLLGaussianLoss, NLLBerGammaLoss
from deep4production.deep.loss.asym import Asym
from deep4production.deep.loss.crps import CRPSSpectralLoss
from deep4production.deep.loss.diffusion import WeightedDenoisingScoreMatchingLoss
from deep4production.deep.loss.classification import (
    BinaryCrossEntropyLoss,
    BernoulliFocalLoss,
)

__all__ = [
    "MaeLoss",
    "MseLoss",
    "QuantisedMSELoss",
    "WeightedMseLoss",
    "NLLGaussianLoss",
    "NLLBerGammaLoss",
    "Asym",
    "CRPSSpectralLoss",
    "WeightedDenoisingScoreMatchingLoss",
    "BinaryCrossEntropyLoss",
    "BernoulliFocalLoss",
]
