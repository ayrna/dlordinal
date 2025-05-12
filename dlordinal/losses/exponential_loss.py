from typing import Optional

import torch
from deprecated.sphinx import deprecated
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module

from dlordinal.soft_labelling import get_exponential_soft_labels

from .custom_targets_loss import CustomTargetsLoss


class ExponentialLoss(CustomTargetsLoss):
    """
    Exponential-regularized loss, as proposed in :footcite:t:`vargas2023exponential`.

    This loss function applies a regularization term based on the Exponential distribution
    to penalize the distance between predicted and true class distributions. It extends
    the `CustomTargetsLoss` by incorporating an Exponential distribution for soft labelling.

    Parameters
    ----------
    base_loss : torch.nn.Module
        The base loss function. It must accept `y_true` as a probability distribution
        (e.g., soft labels or one-hot encoded labels). The base loss is computed between
        the predicted logits (`y_pred`) and the adjusted target labels (`y_true`).

    num_classes : int
        The number of classes (J) in the classification task.

    p : float, default=1.0
        The exponent parameter controlling the shape of the Exponential distribution.
        This parameter influences the steepness of the regularization.

    tau : float, default=1.0
        A scaling parameter for the Exponential distribution that affects the
        regularization term's influence on the target labels.

    eta : float, default=1.0
        A regularization parameter that controls the influence of the regularization term.
        A value of 0 means no regularization, while a value of 1 means the Exponential
        regularization term fully influences the target labels.

    Example
    -------
    >>> import torch
    >>> from dlordinal.losses import ExponentialLoss
    >>> from torch.nn import CrossEntropyLoss
    >>> num_classes = 5
    >>> base_loss = CrossEntropyLoss()
    >>> loss = ExponentialLoss(base_loss, num_classes)
    >>> input = torch.randn(3, num_classes)
    >>> target = torch.randint(0, num_classes, (3,))
    >>> output = loss(input, target)
    """

    def __init__(
        self,
        base_loss: Module,
        num_classes: int,
        p: float = 1.0,
        tau: float = 1.0,
        eta: float = 1.0,
    ):
        # Precompute class probabilities for each label
        cls_probs = torch.tensor(get_exponential_soft_labels(num_classes, p, tau))
        super().__init__(
            base_loss=base_loss,
            cls_probs=cls_probs,
            eta=eta,
        )

    forward = CustomTargetsLoss.forward


# TODO: remove in 3.0.0
@deprecated(
    version="2.4.0",
    reason="Use ExponentialLoss instead with CrossEntropyLoss as base_loss. Will be removed in 3.0.0.",
    category=FutureWarning,
)
class ExponentialCrossEntropyLoss(ExponentialLoss):
    def __init__(
        self,
        num_classes: int,
        eta: float = 1.0,
        p: float = 1,
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
            p=p,
            eta=eta,
        )
