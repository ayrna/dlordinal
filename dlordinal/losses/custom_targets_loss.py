from typing import Optional

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot


class CustomTargetsCrossEntropyLoss(torch.nn.Module):
    """Base class to implement a unimodal regularised cross entropy loss.

    Parameters
    ----------
    cls_probs : Tensor
        The class probabilities tensor.
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
    label_smoothing : float, default=0.0
        Controls the amount of label smoothing for the loss. Zero means no smoothing.
        Default: ``0.0``
    """

    def __init__(
        self,
        cls_probs: Tensor,
        eta: float = 1.0,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.num_classes = cls_probs.size(0)
        self.eta = eta

        self.ce_loss = CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

        # Default class probs initialized to ones
        self.register_buffer("cls_probs", cls_probs.float())

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Method that is called to compute the loss.

        Parameters
        ----------
        input : Tensor
            The input tensor.
        target : Tensor
            The target tensor.

        Returns
        -------
        loss: Tensor
            The computed loss.
        """

        y_prob = self.get_buffer("cls_probs")[target]
        target_oh = one_hot(target, self.num_classes)

        y_true = (1.0 - self.eta) * target_oh + self.eta * y_prob

        return self.ce_loss(input, y_true)
