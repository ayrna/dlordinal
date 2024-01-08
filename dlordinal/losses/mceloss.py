from typing import Optional

import torch
from torch import Tensor


class MCELoss(torch.nn.modules.loss._WeightedLoss):
    """
    Per class mean squared error loss function. Computes the mean squared error for each
    class and reduces it using the specified `reduction`.

    Parameters
    ----------
    num_classes : int
        Number of classes.

    weight : Optional[Tensor], default=None
        A manual rescaling weight given to each class. If given, has to be a Tensor
        of size `J`, where `J` is the number of classes.
        Otherwise, it is treated as if having all ones.

    reduction : str, default='mean'
        Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` |
        ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of
        the output will be divided by the number of elements in the output,
        ``'sum'``: the output will be summed.
    """

    def __init__(
        self,
        num_classes: int,
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
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

    def compute_per_class_mse(self, input: torch.Tensor, target: torch.Tensor):
        """
        Computes the MSE for each class independently.

        Parameters
        ----------
        input : torch.Tensor
            Predicted labels
        target : torch.Tensor
            Ground truth labels

        Returns:
        --------
        mses : torch.Tensor
            MSE values
        """

        if input.shape != target.shape:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes)

        if input.shape != target.shape:
            raise ValueError(
                f"Input shape {input.shape} is not compatible with target shape "
                + f"{target.shape}"
            )

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
            Predicted labels
        target : torch.Tensor
            Ground truth labels

        Returns
        -------
        reduced_mse : torch.Tensor
            MSE per class reduced using the specified `reduction`. If reduction is `none`,
            then a tensor with the MSE for each class is returned.
        """

        input = torch.nn.functional.softmax(input, dim=1)
        target_oh = torch.nn.functional.one_hot(target, num_classes=self.num_classes)

        per_class_mse = self.compute_per_class_mse(input, target_oh)

        if self.reduction == "mean":
            reduced_mse = torch.mean(per_class_mse)
        elif self.reduction == "sum":
            reduced_mse = torch.sum(per_class_mse)
        else:
            reduced_mse = per_class_mse

        return reduced_mse
