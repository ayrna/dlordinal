from typing import Optional

import numpy as np
import torch
from torch.nn.modules.loss import MSELoss


class OrdinalEcocDistanceLoss(torch.nn.Module):
    """Ordinal ECOC distance loss from :footcite:t:`barbero2023error`.

    Parameters
    ----------
    num_classes : int
        Number of classes
    device : torch.device
        Contains the device on which the model is running
    class_weights : Optional[torch.Tensor]
        Contains the weights for each class
    """

    def __init__(
        self, num_classes: int, device, class_weights: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()

        self.target_class = np.ones((num_classes, num_classes - 1), dtype=np.float32)
        self.target_class[np.triu_indices(num_classes, 0, num_classes - 1)] = 0.0
        self.target_class = torch.tensor(
            self.target_class, dtype=torch.float32, device=device, requires_grad=False
        )

        self.class_weights = class_weights

        if self.class_weights is not None:
            assert self.class_weights.shape == (num_classes,)
            self.class_weights = self.class_weights.float().to(device)
            self.mse = MSELoss(reduction="none")
        else:
            self.mse = MSELoss(reduction="sum")

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Parameters
        ----------
        target : torch.Tensor
            Ground truth labels
        input : torch.Tensor
            Predicted labels

        Returns
        -------
        loss : torch.Tensor
            If class_weights is not None, the weighted sum of the MSE loss
            Else the sum of the MSE loss
        """

        target_indices = target.long()

        if self.class_weights is not None:
            target_vector = self.target_class[target_indices]
            weights = self.class_weights[target_indices]  # type: ignore
            return (self.mse(input, target_vector).sum(dim=1) * weights).sum()
        else:
            target_vector = self.target_class[target_indices]
            return self.mse(input, target_vector)
