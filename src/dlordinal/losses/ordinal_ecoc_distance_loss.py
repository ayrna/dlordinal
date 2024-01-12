from typing import Optional

import numpy as np
import torch
from torch import Tensor


class OrdinalECOCDistanceLoss(torch.nn.Module):
    """Ordinal ECOC distance loss from :footcite:t:`barbero2023error` for use
    with :class:`dlordinal.models.OBDECOCModel`. Computes the MSE loss
    between the output of the model (class threshold probabilities) and
    the ideal output vector for each class.

    Parameters
    ----------
    num_classes : int
        Number of classes
    weights : Optional[torch.Tensor]
        Optional weighting for each class
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
            Predicted probabilities for each class threshold: :math:`P(y > q)`
        target : torch.Tensor
            Ground truth labels

        Returns
        -------
        loss : torch.Tensor
        """
        target_vector = self.target_class[target]

        if self.weights is None:
            return self.mse(input, target_vector)
        else:
            weights = self.weights[target]
            return (self.mse(input, target_vector).sum(dim=1) * weights).sum()
