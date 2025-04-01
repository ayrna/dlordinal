from typing import Optional

import torch
from deprecated.sphinx import deprecated
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module

from dlordinal.soft_labelling import get_binomial_soft_labels

from .custom_targets_loss import CustomTargetsLoss


class BinomialLoss(CustomTargetsLoss):
    """Binomial regularised loss from :footcite:t:`liu2020unimodal`.

    Parameters
    ----------
    base_loss: Module
        The base loss function. It must accept y_true as a probability distribution
        (e.g., one-hot or soft labels).
    num_classes : int
        Number of classes.
    eta : float, default=1.0
        Parameter that controls the influence of the regularisation.

    Example
    -------
    >>> import torch
    >>> from dlordinal.losses import BinomialLoss
    >>> from torch.nn import CrossEntropyLoss
    >>> num_classes = 5
    >>> base_loss = CrossEntropyLoss()
    >>> loss = BinomialLoss(base_loss, num_classes)
    >>> input = torch.randn(3, num_classes)
    >>> target = torch.randint(0, num_classes, (3,))
    >>> output = loss(input, target)
    """

    def __init__(
        self,
        base_loss: Module,
        num_classes: int,
        eta: float = 1.0,
    ):
        # Precompute class probabilities for each label
        cls_probs = torch.tensor(get_binomial_soft_labels(num_classes)).float()
        super().__init__(
            base_loss=base_loss,
            cls_probs=cls_probs,
            eta=eta,
        )


# TODO: remove in 3.0.0
@deprecated(
    version="2.4.0",
    reason="Use BinomialLoss instead with CrossEntropyLoss as base_loss. Will be removed in 3.0.0.",
    category=FutureWarning,
)
class BinomialCrossEntropyLoss(BinomialLoss):
    def __init__(
        self,
        num_classes: int,
        eta: float = 1.0,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
    ):
        base_loss = CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
        )
        super().__init__(
            base_loss=base_loss,
            num_classes=num_classes,
            eta=eta,
        )
