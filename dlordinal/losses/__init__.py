from .losses import (BetaCrossEntropyLoss, BinomialCrossEntropyLoss,
                     ExponentialRegularisedCrossEntropyLoss, PoissonCrossEntropyLoss,
                     UnimodalCrossEntropyLoss, WKLoss, OrdinalEcocDistanceLoss)

__all__ = [
    'UnimodalCrossEntropyLoss',
    'BetaCrossEntropyLoss',
    'PoissonCrossEntropyLoss',
    'BinomialCrossEntropyLoss',
    'ExponentialRegularisedCrossEntropyLoss',
    'WKLoss',
    'OrdinalEcocDistanceLoss'
]
