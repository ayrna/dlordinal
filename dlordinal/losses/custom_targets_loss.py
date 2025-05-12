from typing import Optional

import torch
from deprecated.sphinx import deprecated
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.nn.functional import one_hot


class CustomTargetsLoss(torch.nn.Module):
    """
    Base class for implementing a soft labelling loss using class-dependent target smoothing.

    This loss modifies the hard class labels by combining one-hot encoding with prior
    class probabilities. The result is a soft target distribution used as input to a
    base loss function that supports probabilistic targets (e.g., KL divergence or
    soft cross-entropy).

    The smoothing is controlled by the `eta` parameter, where `eta=0` corresponds to
    standard one-hot labels and `eta=1` corresponds to using only the prior class
    probabilities.

    Parameters
    ----------
    base_loss : torch.nn.Module
        The base loss function to apply between predictions and soft targets.
        It must accept `y_true` as a tensor of probabilities, not class indices.
        Specifically, `y_true` should be a vector of probabilities or a one-hot encoded
        vector, where each element represents the probability of the corresponding class

    cls_probs : torch.Tensor
        A tensor of shape (J, J), where each row `j` corresponds to a class-conditional
        target distribution for class `j`. This is used to create the soft targets.

    eta : float, default=1.0
        A scalar in [0, 1] controlling the degree of smoothing applied to the targets.
        Higher values increase the influence of the prior class distributions.

    Example
    -------
    >>> import torch
    >>> import torch.nn as nn
    >>> from dlordinal.losses import CustomTargetsLoss
    >>> base_loss_fn = nn.CrossEntropyLoss()
    >>> cls_probs = torch.tensor([[0.9, 0.075, 0.025], [0.1, 0.6, 0.3], [0.05, 0.15, 0.8]])
    >>> custom_loss_fn = CustomTargetsLoss(base_loss=base_loss_fn, cls_probs=cls_probs, eta=0.5)
    >>> y_pred = torch.randn(2, 3)
    >>> y_true = torch.tensor([0, 2])
    >>> loss = custom_loss_fn(y_pred, y_true)
    >>> print(loss)
    """

    def __init__(
        self,
        base_loss: Module,
        cls_probs: Tensor,
        eta: float = 1.0,
    ):
        super().__init__()

        self.base_loss = base_loss
        self.num_classes = cls_probs.size(0)
        self.eta = eta

        # Default class probs initialized to ones
        self.register_buffer("cls_probs", cls_probs.float())

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Computes the loss between the input predictions and the target labels.

        Parameters
        ----------
        input : torch.Tensor
            A float tensor of shape (N, J) containing predicted logits or probabilities,
            where N is the batch size and J is the number of classes. The expected format
            (logits vs probabilities) depends on the specific base loss function.

        target : torch.Tensor
            An integer tensor of shape (N,) containing the class indices (0 ≤ target < J)
            corresponding to the correct classes for each sample.

        Returns
        -------
        torch.Tensor
            A scalar tensor representing the computed loss.
        """

        y_prob = self.get_buffer("cls_probs")[target]
        target_oh = one_hot(target, self.num_classes)

        y_true = (1.0 - self.eta) * target_oh + self.eta * y_prob

        return self.base_loss(input, y_true)


# TODO: remove in 3.0.0
@deprecated(
    version="2.4.0",
    reason="Use CustomTargetsLoss instead with CrossEntropyLoss as base_loss. Will be removed in 3.0.0.",
    category=FutureWarning,
)
class CustomTargetsCrossEntropyLoss(CustomTargetsLoss):
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
        base_loss = CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
        )
        super().__init__(
            base_loss=base_loss,
            cls_probs=cls_probs,
            eta=eta,
        )
