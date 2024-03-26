from typing import Optional

import torch
from torch import Tensor

from ..distributions import get_exponential_softlabels
from .custom_targets_loss import CustomTargetsCrossEntropyLoss


class ExponentialRegularisedCrossEntropyLoss(CustomTargetsCrossEntropyLoss):
    """Expontential unimodal regularised cross entropy loss from :footcite:t:`liu2020unimodal`.

    Parameters
    ----------
    num_classes : int
        Number of classes.
    eta : float, default=1.0
        Parameter that controls the influence of the regularisation.
    p : float, default=1
        Exponent parameter. Introduced in :footcite:t:`vargas2023exponential` as an
        application of the :math:`L^p` norm.
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
    label_smoothing : float, default=0.0
        Controls the amount of label smoothing for the loss. Zero means no smoothing.
        Default: ``0.0``
    """

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
        label_smoothing: float = 0.0,
    ):
        # Precompute class probabilities for each label
        cls_probs = torch.tensor(get_exponential_softlabels(num_classes, p)).float()

        super().__init__(
            cls_probs=cls_probs,
            eta=eta,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
