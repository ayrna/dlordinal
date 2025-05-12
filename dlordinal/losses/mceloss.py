from typing import Optional

import torch
from torch import Tensor


class MCELoss(torch.nn.modules.loss._WeightedLoss):
    """
    Mean Squared Error (MSE) loss computed per class. This loss function calculates the
    MSE for each class independently and then reduces it based on the specified `reduction`
    method. It is useful in scenarios where each class needs to be treated independently
    during the loss computation.

    Parameters
    ----------
    num_classes : int
        The number of classes in the classification problem.

    weight : Optional[Tensor], default=None
        A tensor of size `J`, where `J` is the number of classes, representing the weight
        for each class. If provided, each class's MSE will be scaled by its corresponding
        weight. If not provided, all classes are treated with equal weight (i.e., all weights
        are set to 1).

    reduction : str, default='mean'
        The method to reduce the MSE values across all classes:
        - `'none'`: No reduction is applied. A tensor of MSE values for each class is returned.
        - `'mean'`: The mean of the MSE values across all classes is returned.
        - `'sum'`: The sum of the MSE values across all classes is returned.

    use_logits : bool, default=False
        If True, the `input` tensor (predictions) is assumed to be in logits format. If False,
        the `input` tensor is treated as probabilities.

    Example
    -------
    >>> import torch
    >>> from torch.nn import CrossEntropyLoss
    >>> from dlordinal.losses import MCELoss
    >>> num_classes = 5
    >>> base_loss = CrossEntropyLoss()
    >>> loss = MCELoss(num_classes=num_classes)
    >>> input = torch.randn(3, num_classes)
    >>> target = torch.randint(0, num_classes, (3,))
    >>> output = loss(input, target)

    Notes
    -----
    - The class supports both the use of logits and probabilities in the predictions.
    - When `use_logits=True`, the input is passed through a softmax function before computing
      the MSE. If `use_logits=False`, the `input` tensor is expected to already contain
      probabilities.
    """

    def __init__(
        self,
        num_classes: int,
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
        use_logits=False,
    ) -> None:
        super().__init__(
            weight=weight, size_average=None, reduce=None, reduction=reduction
        )

        self.num_classes = num_classes

        if weight is not None and weight.shape != (num_classes,):
            raise ValueError(
                f"Weight shape {weight.shape} is not compatible"
                + "with num_classes {num_classes}"
            )

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Reduction {reduction} is not supported."
                + " Please use 'mean', 'sum' or 'none'"
            )

        self.use_logits = use_logits

    def compute_per_class_mse(self, input: torch.Tensor, target: torch.Tensor):
        """
        Computes the mean squared error (MSE) for each class independently.

        Parameters
        ----------
        input : torch.Tensor
            Predicted labels (either logits or probabilities, depending on `use_logits`).

        target : torch.Tensor
            Ground truth labels in one-hot encoding format.

        Returns
        -------
        mses : torch.Tensor
            A tensor containing the MSE values for each class.
        """

        if input.shape != target.shape:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes)

        if input.shape != target.shape:
            raise ValueError(
                f"Input shape {input.shape} is not compatible with target shape "
                + f"{target.shape}"
            )

        if self.use_logits:
            input = torch.nn.functional.softmax(input, dim=1)

        # Compute the squared error for each class
        per_class_se = torch.pow(target - input, 2)

        # Apply class weights if defined
        if self.weight is not None:
            tiled_weight = torch.tile(self.weight, (per_class_se.shape[0], 1))
            per_class_se = per_class_se * tiled_weight

        # Compute the mean squared error for each class
        per_class_mse = torch.mean(per_class_se, dim=0)

        return per_class_mse

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Parameters
        ----------
        input : torch.Tensor
            Predicted labels. Should be logits if `use_logits` is True, otherwise
            probabilities.

        target : torch.Tensor
            Ground truth labels, typically in class indices.

        Returns
        -------
        reduced_mse : torch.Tensor
            The MSE per class reduced using the specified `reduction` method. If
            `reduction='none'`, the MSE values for each class are returned.
            Otherwise, the MSE is reduced according to the method (`mean`, `sum`).
        """

        target_oh = torch.nn.functional.one_hot(target, num_classes=self.num_classes)

        per_class_mse = self.compute_per_class_mse(input, target_oh)

        if self.reduction == "mean":
            reduced_mse = torch.mean(per_class_mse)
        elif self.reduction == "sum":
            reduced_mse = torch.sum(per_class_mse)
        else:
            reduced_mse = per_class_mse

        return reduced_mse
