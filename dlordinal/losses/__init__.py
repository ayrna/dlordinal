from .losses import (
    BetaCrossEntropyLoss,
    BinomialCrossEntropyLoss,
    CustomTargetsCrossEntropyLoss,
    ExponentialRegularisedCrossEntropyLoss,
    GeneralTriangularCrossEntropyLoss,
    OrdinalEcocDistanceLoss,
    PoissonCrossEntropyLoss,
    TriangularCrossEntropyLoss,
    WKLoss,
)

__all__ = [
    "CustomTargetsCrossEntropyLoss",
    "BetaCrossEntropyLoss",
    "PoissonCrossEntropyLoss",
    "BinomialCrossEntropyLoss",
    "ExponentialRegularisedCrossEntropyLoss",
    "WKLoss",
    "OrdinalEcocDistanceLoss",
    "TriangularCrossEntropyLoss",
    "GeneralTriangularCrossEntropyLoss",
]
