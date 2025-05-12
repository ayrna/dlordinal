from typing import Optional

import torch
import torch.nn as nn


class WKLoss(nn.Module):
    """
    Implements Weighted Kappa Loss, introduced by :footcite:t:`deLaTorre2018kappa`.
    Weighted Kappa is widely used in ordinal classification problems. Its value lies in
    :math:`[0, 2]`, where :math:`2` means the random prediction.

    Parameters
    ----------
    num_classes : int
        The number of unique classes in your dataset.
    penalization_type : str, default='quadratic'
        The penalization method for calculating the Kappa statistics. Valid options are
        ``['linear', 'quadratic']``. Defaults to 'quadratic'.
    epsilon : float, default=1e-10
        Small value added to the denominator division by zero.
    weight : Optional[torch.Tensor], default=None
        Class weights to apply during loss computation. Should be a tensor of size
        `(num_classes,)`. If `None`, equal weight is given to all classes.
    use_logits : bool, default=False
        If `True`, the input `y_pred` is treated as logits. If `False`, `y_pred` is treated
        as probabilities. The behavior of the input `y_pred` affects its expected format
        (logits vs. probabilities).

    Example
    -------
    >>> import torch
    >>> from dlordinal.losses import WKLoss
    >>> num_classes = 5
    >>> y_pred = torch.randn(3, num_classes)  # Predicted logits for 3 samples
    >>> y_true = torch.randint(0, num_classes, (3,))  # Ground truth class indices
    >>> loss_fn = WKLoss(num_classes)
    >>> loss = loss_fn(y_pred, y_true)
    >>> print(loss)
    """

    def __init__(
        self,
        num_classes: int,
        penalization_type: str = "quadratic",
        weight: Optional[torch.Tensor] = None,
        epsilon: Optional[float] = 1e-10,
        use_logits=False,
    ):
        super(WKLoss, self).__init__()
        self.num_classes = num_classes
        if penalization_type == "quadratic":
            self.y_pow = 2
        if penalization_type == "linear":
            self.y_pow = 1

        self.epsilon = epsilon
        self.weight = weight
        self.use_logits = use_logits
        self.first_forward = True

    def forward(self, y_pred, y_true):
        """
        Parameters
        ----------
        y_pred : torch.Tensor
            The model predictions. Shape: `(batch_size, num_classes)`.
            If `use_logits=True`, these should be raw logits (unnormalised scores).
            If `use_logits=False`, these should be probabilities (each row summing to 1).

        y_true : torch.Tensor
            Ground truth labels. Shape:
            - `(batch_size,)` if labels are class indices.
            - `(batch_size, num_classes)` if labels are already one-hot encoded.
            In either case, the tensor will be converted to float internally.

        Returns
        -------
        loss : torch.Tensor
            The Weighted Kappa loss. A scalar tensor representing the weighted disagreement
            between predictions and true labels, normalised by expected disagreement.
        """

        num_classes = self.num_classes

        # Convert to onehot if integer labels are provided
        if y_true.dim() == 1:
            y = torch.eye(num_classes).to(y_true.device)
            y_true = y[y_true]

        y_true = y_true.float()

        if self.first_forward:
            if not self.use_logits and not torch.allclose(
                y_pred.sum(dim=1), torch.tensor(1.0, device=y_pred.device)
            ):
                raise ValueError(
                    "When passing use_logits=False, the input y_pred"
                    " should be probabilities, not logits."
                )
            elif self.use_logits and torch.allclose(
                y_pred.sum(dim=1), torch.tensor(1.0, device=y_pred.device)
            ):
                raise ValueError(
                    "When passing use_logits=True, the input y_pred"
                    " should be logits, not probabilities."
                )
            self.first_forward = False

        if self.use_logits:
            y_pred = torch.nn.functional.softmax(y_pred, dim=1)

        repeat_op = (
            torch.Tensor(list(range(num_classes))).unsqueeze(1).repeat((1, num_classes))
        ).to(y_pred.device)
        repeat_op_sq = torch.square((repeat_op - repeat_op.T))
        weights = repeat_op_sq / ((num_classes - 1) ** 2)

        # Apply class weight
        if self.weight is not None:
            # Repeat weight num_classes times in columns
            tiled_weight = self.weight.repeat((num_classes, 1)).to(y_pred.device)
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
