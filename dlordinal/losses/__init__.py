from .losses import (BetaCrossEntropyLoss, BinomialCrossEntropyLoss,
                     ExponentialCrossEntropyLoss, PoissonCrossEntropyLoss,
                     UnimodalCrossEntropyLoss, WKLoss, MSLoss, OrdinalEcocDistanceLoss)

__all__ = [
    'UnimodalCrossEntropyLoss',
    'BetaCrossEntropyLoss',
    'PoissonCrossEntropyLoss',
    'BinomialCrossEntropyLoss',
    'ExponentialCrossEntropyLoss',
    'WKLoss',
    'MSLoss',
    'OrdinalEcocDistanceLoss'
]
