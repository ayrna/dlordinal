from collections import Counter
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

_SORD_FTYPES = ("max", "norm_max", "log", "norm_log", "division", "norm_division")


class SORDLoss(nn.Module):
    """
    Implements the SORD (Softmax-based Ordinal Regression Distribution) Loss
    from :footcite:t:`diaz2019soft`.

    SORD Loss generates a smooth, ordinally-weighted target distribution
    ('softmax_targets') and applies standard Cross-Entropy Loss (or KL
    Divergence) to the model's prediction. The target distribution is
    based on the distance from the true target and can be further customized
    using proximity measures.

    This loss belongs to the family of ordinal losses designed to penalize
    errors based on the severity of the ordinal distance.

    Parameters
    ----------
    alpha : float
        Scaling factor controlling the 'smoothness' of the softmax target
        distribution. A higher alpha results in a sharper distribution.
    num_classes : int
        The total number of ordinal classes (C).
    train_targets : torch.Tensor
        The target labels from the training dataset, required to compute
        class counts and initialize the proximity matrix (prox_mat).
    prox : bool, default=False
        If True, enables the use of class-frequency-based proximity matrices
        (prox_mat) instead of simple L1 distance.
    ftype : str, default="max"
        Defines the function used to convert the proximity matrix into the
        final penalty (phi). Only used if ``prox`` is True. Options include:
        "max", "norm_max", "log", "norm_log", "division", "norm_division".
    weight : Optional[torch.Tensor], default=None
        Optional class weights of shape [num_classes] to handle class
        imbalance.
    use_logits : bool, default=True
        If True, applies F.log_softmax to the input for numerical stability.
        If False, assumes input is probabilities and applies log(input + 1e-9).

    Attributes
    ----------
    prox_mat : Optional[torch.Tensor]
        The precomputed proximity matrix based on training set class frequencies.
        Used when ``prox`` is True.
    norm_prox_mat : Optional[torch.Tensor]
        The L1-normalized version of ``prox_mat``.
    """

    def __init__(
        self,
        alpha: float,
        num_classes: int,
        train_targets: Tensor,
        prox: bool = False,
        ftype: str = "max",
        weight: Optional[torch.Tensor] = None,
        use_logits: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.prox = prox
        self.ftype = ftype
        self.use_logits = use_logits
        self.train_targets = train_targets
        self.weight = weight

        if self.prox and self.ftype not in _SORD_FTYPES:
            raise ValueError(
                f"Invalid ftype '{self.ftype}'. Expected one of {list(_SORD_FTYPES)}."
            )

        # Initialize proximity matrix if needed
        if self.prox:
            self.class_counts_dict = self._create_classcounts_dict(train_targets)
            missing = [c for c, n in self.class_counts_dict.items() if n == 0]
            if missing:
                raise ValueError(
                    "SORDLoss with prox=True needs every class to appear in"
                    f" train_targets, but classes {missing} are absent. Their"
                    " proximity is -log(0) = inf, which makes the loss NaN."
                )
            prox_mat = create_prox_mat(self.class_counts_dict, inv=False).float()
            # Non-persistent buffers follow the module across devices without
            # changing its state_dict.
            self.register_buffer("prox_mat", prox_mat, persistent=False)
            self.register_buffer(
                "norm_prox_mat", F.normalize(prox_mat, p=1, dim=0), persistent=False
            )

    def _create_classcounts_dict(self, targets):
        class_counts = Counter(np.asarray(targets))
        class_counts_dict = {i: class_counts.get(i, 0) for i in range(self.num_classes)}
        return class_counts_dict

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Calculates the SORD loss between the model's prediction and the ordinal
        target distribution.

        Parameters
        ----------
        input : torch.Tensor
            The model's output (logits or probabilities) with shape [Batch, C].
        target : torch.Tensor
            The true ordinal labels with shape [Batch].

        Returns
        -------
        torch.Tensor
            The scalar mean value of the SORD loss.
        """

        if self.use_logits:
            input_logprob = F.log_softmax(input, dim=1)
        else:
            input_logprob = torch.log(input + 1e-9)

        # float32 (float64 only for float64 inputs): float64 is very slow on most GPUs
        work_dtype = torch.promote_types(input_logprob.dtype, torch.float32)

        if not self.prox:
            phi = torch.abs(
                torch.arange(self.num_classes, device=input.device, dtype=work_dtype)
                - target.to(work_dtype).view(-1, 1)
            )
        else:
            # A no-op when the loss was moved with .to(); needed when it was not
            prox_mat = self.prox_mat.to(device=input.device, dtype=work_dtype)
            norm_prox_mat = self.norm_prox_mat.to(device=input.device, dtype=work_dtype)
            if self.ftype == "max":
                phi = torch.max(prox_mat) - prox_mat[target]
            elif self.ftype == "norm_max":
                phi = torch.max(norm_prox_mat) - norm_prox_mat[target]
            elif self.ftype == "norm_log":
                phi = -torch.log(norm_prox_mat[target])
            elif self.ftype == "log":
                phi = -torch.log(prox_mat[target])
            elif self.ftype == "norm_division":
                phi = 1.0 / (norm_prox_mat[target])
            elif self.ftype == "division":
                phi = 1.0 / (prox_mat[target])

        softmax_targets = F.softmax(-self.alpha * phi, dim=1)

        per_sample_loss = -torch.sum(softmax_targets * input_logprob, dim=1)  # [batch]

        # Class weight
        if self.weight is not None:
            # self.weight: [num_classes]
            # target: [batch]
            sample_weights = self.weight[target].to(input.device)  # [batch]
            per_sample_loss = per_sample_loss * sample_weights

        return per_sample_loss.mean()


def create_prox_mat(dist_dict, inv=True):
    """
    Creates a proximity matrix based on class frequency distributions.

    This matrix captures how "close" two classes are based on the frequency
    of classes falling between them in the training data.

    Parameters
    ----------
    dist_dict : dict
        A dictionary containing class indices as keys and their counts/frequencies
        in the training set as values.
    inv : bool, default=True
        If True, the matrix values are calculated as the inverse of the
        logarithm of the normalized cumulative count. If False, the values
        are calculated as the negative logarithm of the normalized cumulative
        count (similar to self-entropy, where distance increases with
        cumulative frequency).

    Returns
    -------
    torch.Tensor
        The proximity matrix of shape [num_classes, num_classes].
    """

    labels = list(dist_dict.keys())
    labels.sort()
    denominator = sum(dist_dict.values())
    prox_mat = np.zeros([len(labels), len(labels)])
    for label1 in labels:
        for label2 in labels:
            label1 = int(label1)
            label2 = int(label2)
            minlabel, maxlabel = min(label1, label2), max(label1, label2)
            numerator = dist_dict[label1] / 2
            if minlabel == label1:  # Above the diagonal
                for tmp_label in range(minlabel + 1, maxlabel + 1):
                    numerator += dist_dict[tmp_label]
            else:  # Under the diagonal
                for tmp_label in range(maxlabel - 1, minlabel - 1, -1):
                    numerator += dist_dict[tmp_label]
            if inv:
                prox_mat[label1][label2] = (-np.log(numerator / denominator)) ** -1
            else:
                prox_mat[label1][label2] = -np.log(numerator / denominator)
    return torch.tensor(prox_mat)
