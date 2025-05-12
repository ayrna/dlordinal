from typing import Optional

import torch
from deprecated.sphinx import deprecated
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module

from ..soft_labelling import get_poisson_soft_labels
from .custom_targets_loss import CustomTargetsLoss


class PoissonLoss(CustomTargetsLoss):
    """
    Poisson unimodal regularised cross-entropy loss from :footcite:t:`liu2020unimodal`.

    This loss combines a base loss function (typically cross-entropy) with a Poisson
    regularisation term to improve classification performance in certain tasks, as described
    in the referenced paper. The base loss is applied to the probability distribution of
    the target labels, and the Poisson regularisation encourages the model to produce more
    balanced probability distributions.

    Parameters
    ----------
    base_loss : torch.nn.Module
        The base loss function (e.g., `CrossEntropyLoss`). It must accept `y_true` as a
        probability distribution (e.g., one-hot encoded or soft labels).
    num_classes : int
        Number of classes (i.e., the size of the probability distribution over the classes).
    eta : float, default=1.0
        Regularisation parameter that controls the influence of the Poisson term. A value
        of 1.0 gives equal weight to the base loss and the Poisson regularisation.
        Smaller values reduce the impact of the regularisation.

    Example
    -------
    >>> import torch
    >>> from dlordinal.losses import PoissonLoss
    >>> from torch.nn import CrossEntropyLoss
    >>> num_classes = 5
    >>> base_loss = CrossEntropyLoss()
    >>> loss = PoissonLoss(base_loss, num_classes)
    >>> input = torch.randn(3, num_classes)  # Predicted logits for 3 samples
    >>> target = torch.randint(0, num_classes, (3,))  # Ground truth class indices
    >>> output = loss(input, target)  # Compute the loss
    >>> print(output)
    """

    def __init__(
        self,
        base_loss: Module,
        num_classes: int,
        eta: float = 1.0,
    ):
        # Precompute class probabilities for each label
        cls_probs = torch.tensor(get_poisson_soft_labels(num_classes)).float()

        super().__init__(
            base_loss=base_loss,
            cls_probs=cls_probs,
            eta=eta,
        )

    forward = CustomTargetsLoss.forward


# TODO: remove in 3.0.0
@deprecated(
    version="2.4.0",
    reason="Use PoissonLoss instead with CrossEntropyLoss as base_loss. Will be removed in 3.0.0.",
    category=FutureWarning,
)
class PoissonCrossEntropyLoss(PoissonLoss):
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
