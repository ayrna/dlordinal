from typing import Optional

import torch
from deprecated.sphinx import deprecated
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module

from dlordinal.soft_labelling import get_triangular_soft_labels

from .custom_targets_loss import CustomTargetsLoss


class TriangularLoss(CustomTargetsLoss):
    """Triangular regularised loss from :footcite:t:`vargas2023softlabelling`.

    This loss function combines a base loss function (such as cross-entropy) with
    a triangular regularisation term, which distributes probabilities to adjacent
    classes. The parameter `alpha2` controls the amount of probability deposited
    into adjacent classes, and `eta` controls the strength of the regularisation.

    Parameters
    ----------
    base_loss : torch.nn.Module
        The base loss function (e.g., `CrossEntropyLoss`). It must accept `y_true`
        as a probability distribution (e.g., one-hot or soft labels).
    num_classes : int
        Number of classes. This defines the size of the probability distribution.
    alpha2 : float, default=0.05
        Parameter that controls the amount of probability deposited in adjacent classes.
        Higher values increase the contribution of adjacent classes.
    eta : float, default=1.0
        Regularisation parameter that controls the influence of the triangular regularisation
        term. A value of 1.0 gives equal weight to the base loss and the triangular term,
        while smaller values reduce the regularisation strength.

    Example
    -------
    >>> import torch
    >>> from dlordinal.losses import TriangularLoss
    >>> from torch.nn import CrossEntropyLoss
    >>> num_classes = 5
    >>> base_loss = CrossEntropyLoss()
    >>> loss = TriangularLoss(base_loss, num_classes)
    >>> input = torch.randn(3, num_classes)  # Predicted logits for 3 samples
    >>> target = torch.randint(0, num_classes, (3,))  # Ground truth class indices
    >>> output = loss(input, target)  # Compute the loss
    >>> print(output)
    """

    def __init__(
        self,
        base_loss: Module,
        num_classes: int,
        alpha2: float = 0.05,
        eta: float = 1.0,
    ):
        # Precompute class probabilities for each label
        cls_probs = torch.tensor(get_triangular_soft_labels(num_classes, alpha2))
        super().__init__(
            base_loss=base_loss,
            cls_probs=cls_probs,
            eta=eta,
        )

    forward = CustomTargetsLoss.forward


# TODO: remove in 3.0.0
@deprecated(
    version="2.4.0",
    reason="Use TriangularLoss instead with CrossEntropyLoss as base_loss. Will be removed in 3.0.0.",
    category=FutureWarning,
)
class TriangularCrossEntropyLoss(TriangularLoss):
    def __init__(
        self,
        num_classes: int,
        alpha2: float = 0.05,
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
            alpha2=alpha2,
            eta=eta,
        )
