from typing import Optional

import numpy as np
import torch
from deprecated.sphinx import deprecated
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module

from ..soft_labelling import get_general_triangular_soft_labels
from .custom_targets_loss import CustomTargetsLoss


class GeneralTriangularLoss(CustomTargetsLoss):
    """
    Generalized triangular loss, as proposed in :footcite:t:`vargas2023gentri`.

    This loss function incorporates a triangular distribution with customizable alpha parameters
    for each class. It applies a regularization term based on the triangular distribution to
    penalize the distance between predicted and true class distributions. The `GeneralTriangularLoss`
    extends `CustomTargetsLoss` by using a generalized triangular distribution for soft labelling.

    Parameters
    ----------
    base_loss : torch.nn.Module
        The base loss function. It must accept `y_true` as a probability distribution
        (e.g., soft labels or one-hot encoded labels). The base loss is computed between
        the predicted logits (`y_pred`) and the adjusted target labels (`y_true`).

    num_classes : int
        The number of classes (J) in the classification task.

    alphas : np.ndarray
        A NumPy array containing the alpha parameters for the triangular distribution.
        The length of this array should be equal to `2 * num_classes`. The alpha parameters control
        the shape of the triangular distribution, influencing the weight given to each class
        in the regularization.

    eta : float, default=1.0
        A regularization parameter that controls the influence of the regularization term.
        A value of 0 means no regularization, while a value of 1 means the triangular
        regularization term fully influences the target labels.

    Example
    -------
    >>> import torch
    >>> from dlordinal.losses import GeneralTriangularLoss
    >>> from torch.nn import CrossEntropyLoss
    >>> import numpy as np
    >>> num_classes = 5
    >>> alphas = np.array([0.1, 0.15, 0.1, 0.05, 0.05, 0.1, 0.15, 0.1, 0.05, 0.05])
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

    forward = CustomTargetsLoss.forward


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
