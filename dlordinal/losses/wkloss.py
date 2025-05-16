from typing import Optional

import torch
import torch.nn as nn


class WKLoss(nn.Module):
    """
    Implements Weighted Kappa Loss, introduced by :footcite:t:`deLaTorre2018kappa`.
    Weighted Kappa is widely used in ordinal classification problems. Its value lies in
    :math:`[0, 2]`, where :math:`2` means the random prediction.

    The loss is computed as follows:

    .. math::
        \\mathcal{L}(X, \\mathbf{y}) =
        \\frac{\\sum\\limits_{i=1}^J \\sum\\limits_{j=1}^J \\omega_{i,j}
        \\sum\\limits_{k=1}^N q_{k,i} ~ p_{y_k,j}}
        {\\frac{1}{N}\\sum\\limits_{i=1}^J \\sum\\limits_{j=1}^J \\omega_{i,j}
        \\left( \\sum\\limits_{k=1}^N q_{k,i} \\right)
        \\left( \\sum\\limits_{k=1}^N p_{y_k, j} \\right)}

    where :math:`q_{k,j}` denotes the normalised predicted probability, computed as:

    .. math::
        q_{k,j} = \\frac{\\text{P}(\\text{y} = j ~|~ \\mathbf{x}_k)}
        {\\sum\\limits_{i=1}^J \\text{P}(\\text{y} = i ~|~ \\mathbf{x}_k)},

    :math:`p_{y_k,j}` is the :math:`j`-th element of the one-hot encoded true label
    for sample :math:`k`, and :math:`\\omega` is the penalisation matrix, defined
    either linearly or quadratically. Its elements are:

    - Linear: :math:`\\omega_{i,j} = \\frac{|i - j|}{J - 1}`
    - Quadratic: :math:`\\omega_{i,j} = \\frac{(i - j)^2}{(J - 1)^2}`

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

    num_classes: int
    penalization_type: str
    weight: Optional[torch.Tensor]
    epsilon: float
    use_logits: bool

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
        self.penalization_type = penalization_type
        self.epsilon = epsilon
        self.weight = weight
        self.use_logits = use_logits
        self.first_forward_ = True

    def _initialize(self, y_pred, y_true):
        # Define error weights matrix
        repeat_op = (
            torch.arange(self.num_classes, device=y_pred.device)
            .unsqueeze(1)
            .expand(self.num_classes, self.num_classes)
        )
        if self.penalization_type == "linear":
            self.weights_ = torch.abs(repeat_op - repeat_op.T) / (self.num_classes - 1)
        elif self.penalization_type == "quadratic":
            self.weights_ = torch.square((repeat_op - repeat_op.T)) / (
                (self.num_classes - 1) ** 2
            )

        # Apply class weight
        if self.weight is not None:
            # Repeat weight num_classes times in columns
            tiled_weight = self.weight.repeat((self.num_classes, 1)).to(y_pred.device)
            self.weights_ *= tiled_weight

    def forward(self, y_pred, y_true):
        """
        Forward pass for the Weighted Kappa loss.

        This method computes the Weighted Kappa loss between the predicted and true labels.
        The loss is based on the weighted disagreement between predictions and true labels,
        normalised by the expected disagreement under independence.

        Parameters
        ----------
        y_pred : torch.Tensor
            The model predictions. Shape: ``(batch_size, num_classes)``.
            If ``use_logits=True``, these should be raw logits (unnormalised scores).
            If ``use_logits=False``, these should be probabilities (rows summing to 1).

        y_true : torch.Tensor
            Ground truth labels.
            Shape:
            - ``(batch_size,)`` if labels are class indices.
            - ``(batch_size, num_classes)`` if already one-hot encoded.
            The tensor will be converted to float internally.

        Returns
        -------
        loss : torch.Tensor
            A scalar tensor representing the weighted disagreement between predictions
            and true labels, normalised by the expected disagreement.
        """

        num_classes = self.num_classes

        # Convert to onehot if integer labels are provided
        if y_true.dim() == 1:
            y = torch.eye(num_classes).to(y_true.device)
            y_true = y[y_true]

        y_true = y_true.float()

        if self.first_forward_:
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

            self._initialize(y_pred, y_true)
            self.first_forward_ = False

        if self.use_logits:
            y_pred = torch.nn.functional.softmax(y_pred, dim=1)

        pred_norm = y_pred / (
            self.epsilon + torch.reshape(torch.sum(y_pred, 1), [-1, 1])
        )

        hist_rater_a = torch.sum(pred_norm, 0)
        hist_rater_b = torch.sum(y_true, 0)

        conf_mat = torch.matmul(pred_norm.T, y_true)

        bsize = y_pred.size(0)
        nom = torch.sum(self.weights_ * conf_mat)
        expected_probs = torch.matmul(
            torch.reshape(hist_rater_a, [num_classes, 1]),
            torch.reshape(hist_rater_b, [1, num_classes]),
        )
        denom = torch.sum(self.weights_ * expected_probs / bsize)

        return nom / (denom + self.epsilon)
