from typing import Optional

import numpy as np
import torch
from torch import Tensor


class OrdinalECOCDistanceLoss(torch.nn.Module):
    """Ordinal ECOC distance loss from :footcite:t:`barbero2023error` for use
    with :class:`dlordinal.wrappers.OBDECOCModel`. Computes the MSE loss
    between the output of the model (class threshold probabilities) and
    the ideal output vector for each class.

    Parameters
    ----------
    num_classes : int
        Number of classes.
    weights : Optional[torch.Tensor]
        Optional weighting for each class. Should be of shape (num_classes,) if provided.

    Attributes
    ----------
    target_class : torch.Tensor
        A tensor of shape (num_classes, num_classes-1) containing the ideal output vectors
        for each class.
    weights : Optional[torch.Tensor]
        A tensor of shape (num_classes,) containing the class-specific weights.
    """

    target_class: Tensor
    weights: Optional[Tensor]

    def __init__(self, num_classes: int, weights: Optional[Tensor] = None) -> None:
        super().__init__()
        target_class = np.ones((num_classes, num_classes - 1), dtype=np.float32)
        target_class[np.triu_indices(num_classes, 0, num_classes - 1)] = 0.0
        target_class = torch.tensor(target_class, dtype=torch.float32)
        self.register_buffer("target_class", target_class)
        self.mse = torch.nn.MSELoss(reduction="sum" if weights is None else "none")
        if weights is not None:
            self.register_buffer("weights", weights)
        else:
            self.weights = None

    def forward(self, input, target):
        """
        Parameters
        ----------
        input : torch.Tensor
            Predicted probabilities for each class threshold, with shape
            (batch_size, num_classes - 1).
        target : torch.Tensor
            Ground truth labels of shape (batch_size,). The labels are integer class indices
            in the range [0, num_classes-1].

        Returns
        -------
        loss : torch.Tensor
            A scalar tensor representing the computed loss. If `weights` is None, the loss is
            computed as the sum of the MSE between `input` and the target vector for each class.
            If `weights` is provided, the loss is computed as the weighted sum of the per-sample
            MSE losses.
        """

        target_vector = self.target_class[target]

        if self.weights is None:
            return self.mse(input, target_vector)
        else:
            weights = self.weights[target]
            return (self.mse(input, target_vector).sum(dim=1) * weights).sum()
