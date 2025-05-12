import torch
import torch.nn.functional as F
from torch import nn


class EMDLoss(nn.Module):
    """
    Computes the squared Earth Mover's Distance (EMD) loss, also known as the
    Ranked Probability Score (RPS), for ordinal classification tasks.

    This implementation follows the formulation presented by :footcite:t:`houys16semd`.
    The squared EMD loss is equivalent to the RPS described in
    :footcite:t:`epstein1969scoring`. It serves as a proper scoring rule for ordinal
    outcomes, encouraging probabilistic predictions that are both accurate and calibrated.

    Errors farther from the true class are penalised more heavily, reflecting the ordinal
    structure of the target variable.

    Parameters
    ----------
    num_classes : int
        The number of ordinal classes (denoted as J).

    Examples
    --------
    >>> import torch
    >>> from dlordinal.losses import EMDLoss
    >>> loss_fn = EMDLoss(num_classes=5)
    >>> y_pred = torch.randn(8, 5)  # Predicted logits
    >>> y_true = torch.tensor([0, 1, 2, 3, 4, 3, 1, 0])  # Class indices
    >>> loss = loss_fn(y_pred, y_true)
    """

    def __init__(self, num_classes: int):
        super(EMDLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        """
        Computes the squared Earth Mover's Distance (Ranked Probability Score) between
        predictions and targets.

        Parameters
        ----------
        y_pred : torch.Tensor
            A tensor of shape (N, J) containing predicted logits, where N is the batch size
            and J is the number of classes.

        y_true : torch.Tensor
            A tensor of shape (N,) containing the true class indices as integers in the
            range [0, J - 1].

        Returns
        -------
        torch.Tensor
            A scalar tensor representing the mean squared EMD loss over the batch.
        """

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
