import torch
import torch.nn.functional as F
from torch import nn


class EMDLoss(nn.Module):
    """
    This implementation of the squared Earth Mover's Distance (EMD) loss follows the formulation presented
    by :footcite:t:`houys16semd`. In this context,
    the squared EMD loss is equivalent to the Ranked Probability Score (RPS)
    as described by :footcite:t:`epstein1969scoring`, which serves as a proper scoring rule for ordinal outcomes.
    The RPS incentivizes truthful probability reporting for ordinal outcomes under the assumption of
    unimodal predictive probabilities, where errors that are farther away are penalized more heavily.

    Parameters
    ----------
    num_classes : int
        Number of classes.
    """

    def __init__(self, num_classes: int):
        super(EMDLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        # One-hot encode true labels
        y_true_one_hot = F.one_hot(y_true, num_classes=self.num_classes)

        # Convert logits to probabilities
        y_pred_proba = torch.nn.functional.softmax(y_pred, dim=1)

        # Compute the CDFs
        pred_cdf = torch.cumsum(y_pred_proba, dim=1)
        true_cdf = torch.cumsum(y_true_one_hot, dim=1)

        # Compute the squared EMD
        emd = torch.sum((pred_cdf - true_cdf) ** 2, dim=1)
        return emd.mean()
