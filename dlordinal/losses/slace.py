from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SLACELoss(nn.Module):
    """
    Implements the SLACE (Soft Labels Accumulating Cross Entropy) loss from
    :footcite:t:`nachmani2025slace`.

    Ordinal regression classifies objects to classes with a natural order,
    where the severity of prediction errors varies (e.g., classifying 'No
    Risk' as 'Critical Risk' is worse than 'High Risk').

    SLACE is ordinality-aware loss designed to ensure the model's
    output is as close as possible to the correct class, considering the
    order of labels.

    It provably satisfies two key properties for ordinal losses:
    **monotonicity** and **balance sensitivity**.

    The mechanism involves generating a smooth, ordinally-weighted target
    probability distribution ('softmax_targets') and applying cross-entropy
    to an accumulated version of the model's predicted distribution
    ('accumulating_softmax').

    Parameters
    ----------
    alpha : float
        Scaling factor controlling the 'smoothness' of the softmax target
        distribution. A higher alpha results in a sharper distribution.
    num_classes : int
        The total number of ordinal classes (C).
    weight : Optional[torch.Tensor], default=None
        Optional class weights of shape [num_classes] to handle class imbalance.
    use_logits : bool, default=True
        If True, assumes 'input' contains logits and applies softmax internally.
        If False, assumes 'input' is already probabilities.

    Attributes
    ----------
    prox_dom : Optional[torch.Tensor]
        The precomputed ordinal dominance matrix used for probability accumulation.
        Registered as a buffer.

    """

    prox_dom: Optional[Tensor]

    def __init__(
        self,
        alpha: float,
        num_classes: int,
        weight: Optional[torch.Tensor] = None,
        use_logits: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.use_logits = use_logits

        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

        # Precompute prox_dom
        labels = torch.arange(self.num_classes)
        h = labels.view(-1, 1, 1)
        i = labels.view(1, -1, 1)
        j = labels.view(1, 1, -1)

        distance_i = torch.abs(i - h)
        distance_j = torch.abs(j - h)

        self.register_buffer("prox_dom", (distance_j <= distance_i).float())

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Calculates the SLACE loss between the model's prediction and the ordinal
        target distribution.

        Parameters
        ----------
        input : torch.Tensor
            The model's output (logits or probabilities) with shape [Batch, num_classes].
        target : torch.Tensor
            The true ordinal labels with shape [Batch] or [Batch, 1].

        Returns
        -------
        torch.Tensor
            The scalar mean value of the SLACE loss.
        """

        if self.use_logits:
            input = F.softmax(input, dim=1)

        # In case target has shape [Batch, 1], flatten to [Batch]
        target = target.view(-1)

        phi = torch.abs(
            torch.arange(self.num_classes, device=input.device).view(1, -1)
            - target.double().view(-1, 1)
        )

        softmax_targets = F.softmax(-self.alpha * phi, dim=1)

        # This is the original formulation by the authors of the paper:
        # one_hot_target = F.one_hot(target, num_classes=self.num_classes).to(
        #     input.device
        # )
        # one_hot_target_comp = 1 - one_hot_target
        # mass_weights = (
        #     one_hot_target * softmax_targets + one_hot_target_comp * softmax_targets
        # )

        # one_hot_target: x
        # softmax_targets: y
        # one_hot_target_comp: 1 - x
        # mass_weights: x*y + (1-x)*y = y
        # Therefore, mass_weights == softmax_targets
        mass_weights = softmax_targets

        accumulating_softmax = (
            torch.matmul(
                self.prox_dom[target.long()].double(),
                torch.unsqueeze(input, 2).double(),
            )
            .double()
            .squeeze(dim=2)
        )

        per_sample_loss = -torch.sum(
            mass_weights * torch.log(accumulating_softmax + 1e-9), dim=1
        )  # [Batch]

        if self.weight is not None:
            sample_weights = self.weight[target].to(input.device)  # [Batch]
            per_sample_loss = per_sample_loss * sample_weights  # [Batch]

        return per_sample_loss.mean()
