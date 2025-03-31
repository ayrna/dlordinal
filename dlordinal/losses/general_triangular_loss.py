from typing import Optional

import numpy as np
import torch
from deprecated.sphinx import deprecated
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module

from ..soft_labelling import get_general_triangular_soft_labels
from .custom_targets_loss import CustomTargetsLoss


class GeneralTriangularLoss(CustomTargetsLoss):
    """Generalised triangular loss from :footcite:t:`vargas2023gentri`.

    Parameters
    ----------
    base_loss: Module
        The base loss function. It must accept y_true as a probability distribution
        (e.g., one-hot or soft labels).
    num_classes : int
        Number of classes.
    alphas : np.ndarray
        The alpha parameters for the triangular distribution.
    eta : float, default=1.0
        Parameter that controls the influence of the regularisation.

    Example
    -------
    >>> import torch
    >>> from dlordinal.losses import GeneralTriangularLoss
    >>> from torch.nn import CrossEntropyLoss
    >>> num_classes = 5
    >>> alphas = np.array([0.1, 0.15, 0.1, 0.05, 0.05])
    >>> base_loss = CrossEntropyLoss()
    >>> loss = GeneralTriangularLoss(base_loss, num_classes, alphas)
    >>> input = torch.randn(3, num_classes)
    >>> target = torch.randint(0, num_classes, (3,))
    >>> output = loss(input, target)
    """

    def __init__(
        self,
        base_loss: Module,
        num_classes: int,
        alphas: np.ndarray,
        eta: float = 1.0,
    ):
        # Precompute class probabilities for each label
        r = get_general_triangular_soft_labels(num_classes, alphas, verbose=0)
        cls_probs = torch.tensor(r)

        super().__init__(
            base_loss=base_loss,
            cls_probs=cls_probs,
            eta=eta,
        )


# TODO: remove in 3.0.0
@deprecated(
    version="2.4.0",
    reason="Use GeneralTriangularLoss instead with CrossEntropyLoss as base_loss. Will be removed in 3.0.0.",
    category=FutureWarning,
)
class GeneralTriangularCrossEntropyLoss(GeneralTriangularLoss):
    def __init__(
        self,
        num_classes: int,
        alphas: np.ndarray,
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
            alphas=alphas,
            eta=eta,
        )
