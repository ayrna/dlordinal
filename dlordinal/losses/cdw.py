import torch
from torch import nn


class CDWCELoss(nn.Module):
    """Class Distance Weighted Cross-Entropy Loss proposed in :footcite:t:`polat2022class`.
    It respects the order of the classes and takes the distance of the classes into
    account in calculation of the cost.

    Parameters
    ----------
    num_classes : int
        Number of classes.
    alpha : float, default=0.5
        The exponent of the distance between target and predicted class.
    weight : Tensor, optional, default=None
        Weight applied to each class when computing the loss. It is based on the target
        class. Can be used to mitigate class imbalance.
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
