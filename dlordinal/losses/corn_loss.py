import torch
from torch import nn


class CORNLoss(nn.Module):
    """Rank-consistent ordinal regression (CORN) loss from :footcite:t:`shi2023corn`.

    See the reference implementation `here <https://github.com/Raschka-research-group/coral-pytorch/blob/313482f86f50b58d8beb9fb54652e943b06745ef/coral_pytorch/losses.py#L87-L153>`__.

    Parameters
    ----------
    num_classes : int
        The number of classes (J).

    Note
    ----
        CORN loss expects the output of your network to be of dimension J-1 because class 0
        is predicted implicitly based on the probabilities of subsequent classes.

        CORN loss does not support probabilistic targets.

    Example
    ---
    >>> import torch
    >>> from dlordinal.losses import CORNLoss
    >>> NUM_CLASSES = 5
    >>> loss_fn = CORNLoss(num_classes=NUM_CLASSES)
    >>> y_pred = torch.randn(3, NUM_CLASSES - 1)
    >>> y_true = torch.tensor([0, 3, 1])
    >>> loss = loss_fn(y_pred, y_true)
    >>> print(loss)
    """

    def __init__(self, num_classes):
        super(CORNLoss, self).__init__()
        self.num_classes = num_classes
        self.log_sigmoid = torch.nn.LogSigmoid()

    def forward(self, y_pred, y_true):
        """
        Computes the CORN loss between predicted logits and true labels.

        Parameters
        ----------
        y_pred : torch.Tensor
            A tensor of shape (N, J - 1) containing predicted logits, where N is the batch
            size and J is the number of classes. These logits are typically the raw outputs
            of a neural network before applying a softmax function.

        y_true : torch.Tensor
            A tensor of shape (N,) with integer class indices (for categorical targets).

        Returns
        -------
        torch.Tensor
            A scalar tensor representing the mean loss over the batch. The result is the
            average of the loss values computed for each sample in the batch.
        """
        sets = []
        for i in range(self.num_classes - 1):
            label_mask = y_true > i - 1
            label_tensor = (y_true[label_mask] > i).to(torch.int64)
            sets.append((label_mask, label_tensor))

        num_examples = 0
        losses = 0.0
        for task_index, s in enumerate(sets):
            train_examples = s[0]
            train_labels = s[1]

            if len(train_labels) < 1:
                continue

            num_examples += len(train_labels)
            pred = y_pred[train_examples, task_index]

            loss = -torch.sum(
                self.log_sigmoid(pred) * train_labels
                + (self.log_sigmoid(pred) - pred) * (1 - train_labels)
            )
            losses += loss

        return losses / num_examples
