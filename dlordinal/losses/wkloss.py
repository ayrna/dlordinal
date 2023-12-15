from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WKLoss(nn.Module):
    """
    Implements Weighted Kappa Loss. Weighted Kappa Loss was introduced in the
    [Weighted kappa loss function for multi-class classification
      of ordinal data in deep learning]
      (https://www.sciencedirect.com/science/article/abs/pii/S0167865517301666).
    Weighted Kappa is widely used in Ordinal Classification Problems. The loss
    value lies in $[-\\infty, \\log 2]$, where $\\log 2$ means the random prediction
    Usage: loss_fn = WKLoss(num_classes = NUM_CLASSES)
    """

    def __init__(
        self,
        num_classes: int,
        penalization_type: Optional[str] = "quadratic",
        weight: Optional[torch.Tensor] = None,
        epsilon: Optional[float] = 1e-10,
    ):
        """Creates a `WeightedKappaLoss` instance.
        Args:
          num_classes: Number of unique classes in your dataset.
          penalization_type: (Optional) Weighting to be considered for calculating
            kappa statistics. A valid value is one of
            ['linear', 'quadratic']. Defaults to 'quadratic'.
          epsilon: (Optional) increment to avoid log zero,
            so the loss will be $ \\log(1 - k + \\epsilon) $, where $ k $ lies
            in $ [-1, 1] $. Defaults to 1e-10.
        Raises:
          ValueError: If the value passed for `penalization_type` is invalid
            i.e. not any one of ['linear', 'quadratic']
        """

        super(WKLoss, self).__init__()
        self.num_classes = num_classes
        if penalization_type == "quadratic":
            self.y_pow = 2
        if penalization_type == "linear":
            self.y_pow = 1

        self.epsilon = epsilon
        self.weight = weight

    def forward(self, y_pred, y_true):
        num_classes = self.num_classes
        y = torch.eye(num_classes)
        y_true = y[y_true]

        y_true = y_true.float()

        repeat_op = (
            torch.Tensor(list(range(num_classes))).unsqueeze(1).repeat((1, num_classes))
        )
        repeat_op_sq = torch.square((repeat_op - repeat_op.T))
        weights = repeat_op_sq / ((num_classes - 1) ** 2)

        # Apply class weight
        if self.weight is not None:
            # Repeat weight num_classes times in columns
            tiled_weight = self.weight.repeat((num_classes, 1))
            weights *= tiled_weight

        pred_ = y_pred**self.y_pow
        pred_norm = pred_ / (self.epsilon + torch.reshape(torch.sum(pred_, 1), [-1, 1]))

        hist_rater_a = torch.sum(pred_norm, 0)
        hist_rater_b = torch.sum(y_true, 0)

        conf_mat = torch.matmul(pred_norm.T, y_true)

        bsize = y_pred.size(0)
        nom = torch.sum(weights * conf_mat)
        expected_probs = torch.matmul(
            torch.reshape(hist_rater_a, [num_classes, 1]),
            torch.reshape(hist_rater_b, [1, num_classes]),
        )
        denom = torch.sum(weights * expected_probs / bsize)

        return nom / (denom + self.epsilon)
