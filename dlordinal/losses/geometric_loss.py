from typing import Optional, Union

import torch
from torch import Tensor

from ..soft_labelling import get_geometric_soft_labels
from .custom_targets_loss import CustomTargetsCrossEntropyLoss


class GeometricCrossEntropyLoss(CustomTargetsCrossEntropyLoss):
    """Unimodal label smoothing based on the discrete geometric distribution according to :footcite:t:`haas2023geometric`.

    Parameters
    ----------
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
    weight : Optional[Tensor], default=None
        A manual rescaling weight given to each class. If given, has to be a Tensor
        of size `C`. Otherwise, it is treated as if having all ones.
    size_average : Optional[bool], default=None
        Deprecated (see :attr:`reduction`). By default, the losses are averaged over
        each loss element in the batch. Note that for some losses, there are
        multiple elements per sample. If the field :attr:`size_average` is set to
        ``False``, the losses are instead summed for each minibatch. Ignored when
        reduce is ``False``. Default: ``True``
    ignore_index : int, default=-100
        Specifies a target value that is ignored and does not contribute to the
        input gradient. When :attr:`size_average` is ``True``, the loss is averaged
        over non-ignored targets.
    reduce : Optional[bool], default=None
        Deprecated (see :attr:`reduction`). By default, the losses are averaged or
        summed over observations for each minibatch depending on :attr:`size_average`.
        When :attr:`reduce` is ``False``, returns a loss per batch element instead
        and ignores :attr:`size_average`. Default: ``True``
    reduction : str, default='mean'
        Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` |
        ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of
        the output will be divided by the number of elements in the output,
        ``'sum'``: the output will be summed. Note: :attr:`size_average` and
        :attr:`reduce` are in the process of being deprecated, and in the meantime,
        specifying either of those two args will override :attr:`reduction`.
        Default: ``'mean'``
    """

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
        # Precompute class probabilities for each label
        r = get_geometric_soft_labels(num_classes, alphas)
        cls_probs = torch.tensor(r)

        super().__init__(
            cls_probs=cls_probs,
            eta=eta,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=0.0,
        )
