from typing import Optional

import torch
from torch import Tensor

from .mceloss import MCELoss
from .wkloss import WKLoss


class MCEAndWKLoss(torch.nn.modules.loss._WeightedLoss):
    """
    The loss function integrates both MCELoss and WKLoss, concurrently minimising
    error distances while preventing the omission of classes from predictions.

    Parameters
    ----------
    num_classes : int
        Number of classes.
    C : float, default=0.5
        Weighting factor for WK loss (C) and MCE loss (1-C). Must be between 0 and 1.
    wk_penalization_type : str, default='quadratic'
        The penalization type of WK loss to use ('quadratic' or 'linear').
        See WKLoss for more details.
    weight : Optional[Tensor], default=None
        A manual rescaling weight given to each class. If given, must be a Tensor
        of size `J`, where `J` is the number of classes. Otherwise, it is treated
        as if having all ones.
    reduction : str, default='mean'
        Specifies the reduction to apply to the target: ``'none'`` | ``'mean'`` |
        ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum
        of the target will be divided by the number of elements in the target,
        ``'sum'``: the target will be summed.
    use_logits : bool, default=False
        If True, the `input` will be treated as logits. If False, it will be
        treated as probabilities.

    Example
    -------
    >>> import torch
    >>> from dlordinal.losses import MCEAndWKLoss
    >>> num_classes = 5
    >>> loss = MCEAndWKLoss(num_classes, C=0.7, use_logits=True)
    >>> input = torch.randn(3, num_classes)
    >>> target = torch.randint(0, num_classes, (3,))
    >>> target = loss(input, target)
    """

    def __init__(
        self,
        num_classes: int,
        C: float = 0.5,
        wk_penalization_type: str = "quadratic",
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
        use_logits=False,
    ) -> None:
        super().__init__(
            weight=weight, size_average=None, reduce=None, reduction=reduction
        )

        self.num_classes = num_classes
        self.C = C
        self.wk_penalization_type = wk_penalization_type

        if weight is not None and weight.shape != (num_classes,):
            raise ValueError(
                f"Weight shape {weight.shape} is not compatible"
                + "with num_classes {num_classes}"
            )

        if C < 0 or C > 1:
            raise ValueError(f"C must be between 0 and 1, but is {C}")

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Reduction {reduction} is not supported."
                + " Please use 'mean', 'sum' or 'none'"
            )

        self.use_logits = use_logits

        self.wk = WKLoss(
            self.num_classes,
            penalization_type=self.wk_penalization_type,
            weight=weight,
            use_logits=self.use_logits,
        )
        self.mce = MCELoss(self.num_classes, weight=weight, use_logits=self.use_logits)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Parameters
        ----------
        input : torch.Tensor
            Ground truth labels of shape (N,) where N is the batch size. Values are
            class indices in the range [0, num_classes-1].
        target : torch.Tensor
            Predicted labels of shape (N, num_classes). If `use_logits` is True, these
            are logits. Otherwise, they are probabilities.

        Returns
        -------
        loss : torch.Tensor
            A scalar tensor representing the weighted sum of MCE and QWK loss. If
            `reduction` is 'none', returns a tensor of shape (num_classes,) containing
            the per-class loss for both MCE and WK losses.
        """

        wk_result = self.wk(input, target)
        mce_result = self.mce(input, target)

        return self.C * wk_result + (1 - self.C) * mce_result
