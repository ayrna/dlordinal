import torch
from torch import nn


class CDWCELoss(nn.Module):
    """
    Class Distance Weighted Cross-Entropy Loss, proposed in :footcite:t:`polat2022class`.
    This loss function takes the order of the classes into account by applying a
    distance weighting between the target and predicted classes. The weight applied
    is determined by the distance between the true and predicted classes, controlled
    by the `alpha` parameter.

    This loss function is particularly useful for ordinal classification tasks where
    the order of the classes matters, and penalties should increase as the distance between
    the true and predicted classes grows.

    Parameters
    ----------
    num_classes : int
        The number of classes (J).

    alpha : float, default=0.5
        Exponent that controls the influence of the class distance in the loss calculation.
        A higher `alpha` gives more weight to classes that are farther apart.

    weight : torch.Tensor, optional, default=None
        A tensor of shape (J,) representing class-specific weights, used to address class
        imbalance. The weight for each class is applied during loss computation and can
        be normalised automatically. If `None`, no class weights are applied.

    margin : float, default=0.0
        A margin value that encourages a minimum separation between classes.

    reduction : {'none', 'sum', 'mean', 'weighted_mean'}, default='mean'
        Reduction over samples. The default ``'mean'`` divides the total loss by
        the batch size. When class weights are provided, ``'none'``, ``'sum'`` and
        ``'mean'`` retain their normalisation by the sum of all class weights.
        ``'weighted_mean'`` instead divides the weighted loss by the sum of the
        target class weights in the batch. Without class weights, it equals
        ``'mean'``. A batch with zero total target weight returns zero.
        Weighted means use at least float32 precision for class weights and
        their reduction.

    normalize_distance : bool, default=False
        Divide each sample's class distance weights by their sum before computing
        the loss. This normalisation is independent of ``reduction``.

    Example
    -------
    >>> import torch
    >>> from dlordinal.losses import CDWCELoss
    >>> loss_fn = CDWCELoss(num_classes=5, alpha=1.0)
    >>> y_pred = torch.randn(3, 5)
    >>> y_true = torch.tensor([0, 3, 1])
    >>> loss = loss_fn(y_pred, y_true)
    >>> print(loss)
    """

    def __init__(
        self,
        num_classes,
        alpha=0.5,
        weight=None,
        margin=0.0,
        *,
        reduction="mean",
        normalize_distance=False,
    ):
        super(CDWCELoss, self).__init__()
        if reduction not in ("none", "sum", "mean", "weighted_mean"):
            raise ValueError(
                "reduction must be 'none', 'sum', 'mean' or 'weighted_mean'"
            )
        if reduction == "weighted_mean" and weight is not None:
            if not isinstance(weight, torch.Tensor) or weight.shape != (num_classes,):
                raise ValueError(f"weight must be a tensor of shape ({num_classes},)")
            if (
                weight.is_complex()
                or not torch.isfinite(weight).all()
                or (weight < 0).any()
            ):
                raise ValueError("weight must contain only finite, nonnegative values")
        self.num_classes = num_classes
        self.alpha = alpha
        self.reduction = reduction
        self.normalize_distance = normalize_distance
        self.weight_ = weight
        self.normalised_weight_ = None
        if self.weight_ is not None and reduction != "weighted_mean":
            self.normalised_weight_ = self.weight_ / self.weight_.sum()
        self.margin = margin

    def forward(self, y_pred, y_true):
        """
        Computes the Class Distance Weighted Cross-Entropy loss between predicted logits
        and true labels.

        Parameters
        ----------
        y_pred : torch.Tensor
            A tensor of shape (N, J) containing predicted logits, where N is the batch
            size and J is the number of classes. These logits are typically the raw outputs
            of a neural network before applying a softmax function.

        y_true : torch.Tensor
            A tensor containing the ground-truth labels. It can be either:
            - A tensor of shape (N,) with integer class indices (for categorical targets).
            - A tensor of shape (N, J) with one-hot encoded labels (for probabilistic targets).

        Returns
        -------
        torch.Tensor
            A tensor of shape (N,) when ``reduction='none'``. Otherwise, a scalar
            tensor containing the requested reduction over samples.
        """

        if y_true.dim() > 1:
            y_true_indices = y_true.argmax(dim=1, keepdim=True)
        else:
            y_true_indices = y_true.view(-1, 1)

        N = y_true_indices.size(0)
        J = self.num_classes

        if self.reduction == "weighted_mean" and N == 0:
            return y_pred.sum() * 0.0

        sample_weights = None
        if self.reduction == "weighted_mean" and self.weight_ is not None:
            weight_dtype = torch.promote_types(self.weight_.dtype, y_pred.dtype)
            weight_dtype = torch.promote_types(weight_dtype, torch.float32)
            class_weights = self.weight_.to(device=y_pred.device, dtype=weight_dtype)
            sample_weights = class_weights[y_true_indices]
            # A common factor cancels in the weighted mean. Scaling before the
            # sums avoids overflow with large integer or low precision weights.
            max_weight = sample_weights.max()
            sample_weights = sample_weights / torch.where(
                max_weight > 0, max_weight, torch.ones_like(max_weight)
            )
            # Zero-weight samples have no contribution. Safe logits also avoid
            # propagating undefined intermediate values from ignored samples.
            y_pred = torch.where(sample_weights > 0, y_pred, torch.zeros_like(y_pred))

        s = torch.exp(y_pred).sum(dim=1, keepdim=True)
        if self.margin > 0.0:
            l1 = torch.log(torch.min(s - torch.exp(y_pred) + self.margin * s, s) + 1e-8)
            l2 = torch.log(s + 1e-8)
            l_1_2 = l1 - l2
        else:
            l1 = torch.log(s - torch.exp(y_pred) + 1e-8)
            l2 = torch.log(s + 1e-8)
            l_1_2 = l1 - l2

        i_indices = torch.arange(J).view(1, -1).expand(N, J).to(y_true_indices.device)
        weights = (torch.abs(i_indices - y_true_indices) ** self.alpha).float()
        if self.normalize_distance:
            distance_sum = weights.sum(dim=1, keepdim=True)
            weights = weights / torch.where(
                distance_sum > 0, distance_sum, torch.ones_like(distance_sum)
            )

        loss = l_1_2 * weights

        if self.reduction == "weighted_mean":
            if sample_weights is None:
                return -loss.sum() / N
            sample_weight_sum = sample_weights.sum()
            positive_weight = sample_weight_sum > 0
            denominator = torch.where(
                positive_weight, sample_weight_sum, torch.ones_like(sample_weight_sum)
            )
            weighted_loss = -(loss * sample_weights).sum()
            return torch.where(
                positive_weight, weighted_loss / denominator, weighted_loss * 0.0
            )

        if self.weight_ is not None and self.normalised_weight_ is not None:
            if self.normalised_weight_.device != loss.device:
                self.normalised_weight_ = self.normalised_weight_.to(loss.device)

            tiled_class_weight = self.normalised_weight_.view(1, -1).expand(N, J)
            sample_weights = torch.gather(
                tiled_class_weight, dim=1, index=y_true_indices
            )
            loss = loss * sample_weights

        if self.reduction == "none":
            return -loss.sum(dim=1)
        if self.reduction == "sum":
            return -loss.sum()

        return -loss.sum() / N
