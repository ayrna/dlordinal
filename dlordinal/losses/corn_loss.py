import torch
from torch import nn


class CORNLoss(nn.Module):
    """Rank-consistent ordinal regression (CORN) loss from :footcite:t:`shi2023corn`.

    See the reference implementation `here <https://github.com/Raschka-research-group/coral-pytorch/blob/313482f86f50b58d8beb9fb54652e943b06745ef/coral_pytorch/losses.py#L87-L153>`__.

    The loss accepts hard targets (class indices) and probabilistic targets (a
    distribution over the classes for every example, for instance a soft label, the
    uniform distribution over a set of acceptable classes, or the empirical distribution
    of several annotators). With probabilistic targets it is the *soft-CORN* loss: the
    hard subset membership :math:`1[y \\geq k]` and the hard indicator :math:`1[y > k]`
    of the conditional tasks are replaced by their expectations :math:`P(y \\geq k)` and
    :math:`P(y > k)` under the target distribution :math:`p`,

    .. math::

        \\mathcal{L} = -\\frac{\\sum_{i}\\sum_{k=0}^{J-2}\\left[ P_i(y > k) \\log\\sigma(z_{ik})
        + p_{ik} \\log(1-\\sigma(z_{ik})) \\right]}{\\sum_{i}\\sum_{k=0}^{J-2} P_i(y \\geq k)},

    which equals the hard CORN loss when every target is one-hot. Up to the normalisation
    it is the cross-entropy between the target distribution and the class probabilities
    that the CORN logits define, :math:`P(y=j) = \\prod_{k<j}\\sigma(z_k)\\,(1-\\sigma(z_j))`,
    so it is a proper scoring rule for the target distribution. A benchmark of this loss
    against other losses for learning from set-valued ordinal labels is available in
    :footcite:t:`colley2026softcorn`.

    Parameters
    ----------
    num_classes : int
        The number of classes (J).

    Note
    ----
        CORN loss expects the output of your network to be of dimension J-1 because class 0
        is predicted implicitly based on the probabilities of subsequent classes.

        Probabilistic targets must be non-negative and sum to one over the J classes.

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

    Probabilistic targets, here the uniform distribution over the accepted classes
    ``{1, 2}``, the single class ``3`` and the classes ``{0, 1, 2}``:

    >>> y_true = torch.tensor(
    ...     [[0.0, 0.5, 0.5, 0.0, 0.0],
    ...      [0.0, 0.0, 0.0, 1.0, 0.0],
    ...      [1 / 3, 1 / 3, 1 / 3, 0.0, 0.0]]
    ... )
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
            Either a tensor of shape (N,) with integer class indices (for categorical
            targets), or a tensor of shape (N, J) with a probability distribution over the
            classes for every example (for probabilistic targets, which gives the
            soft-CORN loss).

        Returns
        -------
        torch.Tensor
            A scalar tensor representing the mean loss over the batch. The result is the
            average of the loss values computed for each sample in the batch.
        """
        if y_true.ndim == 2:
            return self._soft_forward(y_pred, y_true)

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

    def _soft_forward(self, y_pred, y_true):
        """CORN loss for probabilistic targets of shape (N, J) (soft-CORN)."""
        if y_true.shape[1] != self.num_classes:
            raise ValueError(
                "Probabilistic targets must have shape (N, num_classes) = "
                f"(N, {self.num_classes}), got {tuple(y_true.shape)}."
            )
        if y_pred.shape != (y_true.shape[0], self.num_classes - 1):
            raise ValueError(
                "y_pred must have shape (N, num_classes - 1) = "
                f"({y_true.shape[0]}, {self.num_classes - 1}), got {tuple(y_pred.shape)}."
            )

        p = y_true.to(y_pred.dtype)
        # suffix[:, k] = P(y >= k); summing from the top avoids the cancellation of 1 - cumsum
        suffix = torch.flip(torch.cumsum(torch.flip(p, dims=[1]), dim=1), dims=[1])
        # Expected size of the k-th conditional training set, P(y >= k)
        p_ge = suffix[:, :-1]
        # Expected positive part of that set, P(y > k)
        p_gt = suffix[:, 1:]
        # Expected negative part of that set, P(y = k) = P(y >= k) - P(y > k)
        p_eq = p[:, :-1]

        log_sig = self.log_sigmoid(y_pred)
        loss = -torch.sum(p_gt * log_sig + p_eq * (log_sig - y_pred))

        return loss / torch.sum(p_ge)
