from typing import Optional

import torch
from deprecated.sphinx import deprecated
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module

from dlordinal.soft_labelling import get_beta_soft_labels

from .custom_targets_loss import CustomTargetsLoss


class BetaLoss(CustomTargetsLoss):
    """
    Beta-regularized loss, as proposed in :footcite:t:`vargas2022unimodal`.

    This loss function applies a regularization term based on the Beta distribution
    to penalize the distance between predicted and true class distributions. It extends
    the `CustomTargetsLoss` by incorporating a Beta distribution for soft labelling.

    Parameters
    ----------
    base_loss : torch.nn.Module
        The base loss function. It must accept `y_true` as a probability distribution
        (e.g., soft labels or one-hot encoded labels). The base loss is applied
        between the predicted logits (`y_pred`) and the adjusted target labels (`y_true`).

    num_classes : int
        The number of classes (J) in the classification task.

    params_set : str, default='standard'
        The set of parameters used for the Beta distribution. The parameters are chosen
        from the `_beta_params_set` dictionary, which defines how the Beta distribution
        should behave for each class. The "standard" set corresponds to the default Beta
        distribution parameters.

    eta : float, default=1.0
        A regularization parameter that controls the balance between the base loss and
        the regularization term. A value of 0 means no regularization, while a value
        of 1 means the Beta regularization term fully influences the target labels.

    Example
    -------
    >>> import torch
    >>> from dlordinal.losses import BetaLoss
    >>> from torch.nn import CrossEntropyLoss
    >>> num_classes = 5
    >>> base_loss = CrossEntropyLoss()
    >>> loss = BetaLoss(base_loss, num_classes)
    >>> input = torch.randn(3, num_classes)
    >>> target = torch.randint(0, num_classes, (3,))
    >>> output = loss(input, target)
    """

    def __init__(
        self,
        base_loss: Module,
        num_classes: int,
        params_set: str = "standard",
        eta: float = 1.0,
    ):
        # Precompute class probabilities for each label
        cls_probs = torch.tensor(get_beta_soft_labels(num_classes, params_set)).float()
        super().__init__(
            base_loss=base_loss,
            cls_probs=cls_probs,
            eta=eta,
        )

    forward = CustomTargetsLoss.forward


# TODO: remove in 3.0.0
@deprecated(
    version="2.4.0",
    reason="Use BetaLoss instead with CrossEntropyLoss as base_loss. Will be removed in 3.0.0.",
    category=FutureWarning,
)
class BetaCrossEntropyLoss(BetaLoss):
    def __init__(
        self,
        num_classes: int,
        params_set: str = "standard",
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
            params_set=params_set,
            eta=eta,
        )
