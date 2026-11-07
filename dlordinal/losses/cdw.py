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

    def __init__(self, num_classes, alpha=0.5, weight=None, margin=0.0):
        super(CDWCELoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.weight_ = weight
        self.normalised_weight_ = None
        if self.weight_ is not None:
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
            A scalar tensor representing the mean loss over the batch. The result is the
            average of the loss values computed for each sample in the batch.
        """

        if y_true.dim() > 1:
            y_true_indices = y_true.argmax(dim=1, keepdim=True)
        else:
            y_true_indices = y_true.view(-1, 1)

        y_true_indices = y_true_indices.long()
        N = y_true_indices.size(0)
        J = self.num_classes

        if not y_pred.is_floating_point():
            y_pred = y_pred.float()

        # Work with log-probabilities so that large logits cannot overflow exp().
        log_p = torch.nn.functional.log_softmax(y_pred, dim=1)
        p = log_p.exp()
        if self.margin > 0.0:
            # log(min(1 - p + margin, 1)); the argument is at least `margin` > 0
            l_1_2 = torch.log(torch.clamp(1.0 - p + self.margin, max=1.0))
        else:
            # log(1 - p) without cancellation. log1p(-p) is accurate for p <= 0.5,
            # and at most one class per sample has p > 0.5. For that class, 1 - p is
            # the total probability of the other classes, computed as a logsumexp of
            # their log-probabilities. The clamp keeps the branch that torch.where
            # discards finite, so its gradient cannot be NaN.
            dominant = p > 0.5
            log_rest = torch.logsumexp(
                log_p.masked_fill(dominant, float("-inf")), dim=1, keepdim=True
            )
            l_1_2 = torch.where(dominant, log_rest, torch.log1p(-p.clamp(max=0.5)))

        i_indices = torch.arange(J, device=y_true_indices.device).view(1, -1)
        weights = (torch.abs(i_indices - y_true_indices) ** self.alpha).to(l_1_2.dtype)

        loss = l_1_2 * weights

        if self.weight_ is not None and self.normalised_weight_ is not None:
            if self.normalised_weight_.device != loss.device:
                self.normalised_weight_ = self.normalised_weight_.to(loss.device)

            tiled_class_weight = self.normalised_weight_.view(1, -1).expand(N, J)
            sample_weights = torch.gather(
                tiled_class_weight, dim=1, index=y_true_indices
            )
            loss = loss * sample_weights

        loss = loss.sum()

        return -loss / N
