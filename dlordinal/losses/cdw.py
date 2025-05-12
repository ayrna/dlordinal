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

    def __init__(self, num_classes, alpha=0.5, weight=None):
        super(CDWCELoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.weight_ = weight
        self.normalised_weight_ = None
        if self.weight_ is not None:
            self.normalised_weight_ = self.weight_ / self.weight_.sum()

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

        N = y_true_indices.size(0)
        J = self.num_classes

        s = torch.exp(y_pred).sum(dim=1, keepdim=True)
        l1 = torch.log(s - torch.exp(y_pred) + 1e-8)
        l2 = torch.log(s + 1e-8)
        l_1_2 = l1 - l2

        i_indices = torch.arange(J).view(1, -1).expand(N, J).to(y_true_indices.device)
        weights = (torch.abs(i_indices - y_true_indices) ** self.alpha).float()

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
