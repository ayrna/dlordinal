from .beta_loss import BetaCrossEntropyLoss
from .binomial_loss import BinomialCrossEntropyLoss
from .custom_targets_loss import CustomTargetsCrossEntropyLoss
from .exponential_loss import ExponentialRegularisedCrossEntropyLoss
from .general_triangular_loss import GeneralTriangularCrossEntropyLoss
from .mceloss import MCELoss
from .mcewkloss import MCEAndWKLoss
from .ordinal_ecoc_distance_loss import OrdinalECOCDistanceLoss
from .poisson_loss import PoissonCrossEntropyLoss
from .triangular_loss import TriangularCrossEntropyLoss
from .wkloss import WKLoss

__all__ = [
    "CustomTargetsCrossEntropyLoss",
    "BetaCrossEntropyLoss",
    "PoissonCrossEntropyLoss",
    "BinomialCrossEntropyLoss",
    "ExponentialRegularisedCrossEntropyLoss",
    "WKLoss",
    "MCELoss",
    "MCEAndWKLoss",
    "OrdinalECOCDistanceLoss",
    "TriangularCrossEntropyLoss",
    "GeneralTriangularCrossEntropyLoss",
]
