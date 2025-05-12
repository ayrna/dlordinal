from typing import Optional, Union

import torch
from deprecated.sphinx import deprecated
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module

from ..soft_labelling import get_geometric_soft_labels
from .custom_targets_loss import CustomTargetsLoss


class GeometricLoss(CustomTargetsLoss):
    """
    Unimodal label smoothing based on the discrete geometric distribution according
    to :footcite:t:`haas2023geometric`.

    Parameters
    ----------
    base_loss : Module
        The base loss function. It must accept y_true as a probability distribution
        (e.g., one-hot or soft labels).
    num_classes : int
        Number of classes.
    alphas : float or list, default=0.1
        The smoothing factor(s) for geometric distribution-based unimodal smoothing.

        - **Single alpha value**:
          When a single alpha value in the range `[0, 1]`, e.g., `0.1`, is provided,
          all classes will be smoothed equally and symmetrically.
          This is done by deducting alpha from the actual class, :math:`1 - \\alpha`,
          and allocating :math:`\\alpha` to the rest of the classes, decreasing monotonically
          from the actual class in the form of the geometric distribution.

        - **List of alpha values**:
          Alternatively, a list of size :attr:`num_classes` can be provided to specify class-wise symmetric
          smoothing factors. An example for five classes is: `[0.2, 0.05, 0.1, 0.15, 0.1]`.

        - **List of smoothing relations**:
          To control the fraction of the left-over probability mass :math:`\\alpha` allocated to the left
          (:math:`F_l \\in [0,1]`) and right (:math:`F_r \\in [0,1]`) sides of the true class, with
          :math:`F_l + F_r = 1`, a list of smoothing relations of the form :math:`(\\alpha, F_l, F_r)`
          can be specified. This enables asymmetric unimodal smoothing. An example for five classes is:
          `[(0.2, 0.0, 1.0), (0.05, 0.8, 0.2), (0.1, 0.5, 0.5), (0.15, 0.6, 0.4), (0.1, 1.0, 0.0)]`.

    eta : float, default=1.0
        Parameter that controls the influence of the regularisation.

    Example
    -------
    >>> import torch
    >>> from dlordinal.losses import GeometricLoss
    >>> from torch.nn import CrossEntropyLoss
    >>> num_classes = 5
    >>> base_loss = CrossEntropyLoss()
    >>> loss = GeometricLoss(base_loss, num_classes)
    >>> input = torch.randn(3, num_classes)
    >>> target = torch.randint(0, num_classes, (3,))
    >>> output = loss(input, target)
    """

    def __init__(
        self,
        base_loss: Module,
        num_classes: int,
        alphas: Union[float, list] = 0.1,
        eta: float = 1.0,
    ):
        # Precompute class probabilities for each label
        r = get_geometric_soft_labels(num_classes, alphas)
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
    reason="Use GeometricLoss instead with CrossEntropyLoss as base_loss. Will be removed in 3.0.0.",
    category=FutureWarning,
)
class GeometricCrossEntropyLoss(GeometricLoss):
    def __init__(
        self,
        num_classes: int,
        alphas: Union[float, list] = 0.1,
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
