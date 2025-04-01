from .beta_loss import BetaCrossEntropyLoss, BetaLoss
from .binomial_loss import BinomialCrossEntropyLoss, BinomialLoss
from .cdw import CDWCELoss
from .custom_targets_loss import CustomTargetsCrossEntropyLoss, CustomTargetsLoss
from .emd_loss import EMDLoss
from .exponential_loss import ExponentialCrossEntropyLoss, ExponentialLoss
from .general_triangular_loss import (
    GeneralTriangularCrossEntropyLoss,
    GeneralTriangularLoss,
)
from .geometric_loss import GeometricCrossEntropyLoss, GeometricLoss
from .mceloss import MCELoss
from .mcewkloss import MCEAndWKLoss
from .ordinal_ecoc_distance_loss import OrdinalECOCDistanceLoss
from .poisson_loss import PoissonCrossEntropyLoss, PoissonLoss
from .triangular_loss import TriangularCrossEntropyLoss, TriangularLoss
from .wkloss import WKLoss

__all__ = [
    "BetaCrossEntropyLoss",
    "BetaLoss",
    "BinomialCrossEntropyLoss",
    "BinomialLoss",
    "CDWCELoss",
    "CustomTargetsLoss",
    "CustomTargetsCrossEntropyLoss",
    "EMDLoss",
    "ExponentialCrossEntropyLoss",
    "ExponentialLoss",
    "GeneralTriangularCrossEntropyLoss",
    "GeneralTriangularLoss",
    "GeometricCrossEntropyLoss",
    "GeometricLoss",
    "MCEAndWKLoss",
    "MCELoss",
    "OrdinalECOCDistanceLoss",
    "PoissonCrossEntropyLoss",
    "PoissonLoss",
    "TriangularCrossEntropyLoss",
    "TriangularLoss",
    "WKLoss",
]
