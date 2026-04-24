from typing import Callable

import torch
import torch.nn as nn


class GaussianUncertaintyLossWrapper(nn.Module):
    """
    Loss wrapper for models using a Gaussian Uncertainty (GU) output layer.

    This wrapper augments a base loss function with a regularisation term on
    the predicted uncertainty (sigma), encouraging the model to avoid
    unnecessarily large variance estimates.

    The total loss is defined as:

        total_loss = base_loss(probs, y_true)
                     + (1 - alpha) * mean(sigma^2)

    where:
    - `probs` is the predicted discrete probability distribution
    - `sigma` is the predicted standard deviation
    - `alpha` controls the strength of the regularisation

    Parameters
    ----------
    base_loss : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Loss function applied to the predicted probabilities and targets.
        Typically something like `nn.CrossEntropyLoss` (adapted to probabilities)
        or another suitable criterion.

    alpha : float, optional
        Weighting factor between the base loss and the uncertainty penalty.
        Higher values reduce the impact of the sigma regularisation.
        Default is 0.5.

    Attributes
    ----------
    base_loss : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Wrapped loss function.

    alpha : float
        Regularisation weighting factor.

    Notes
    -----
    - The wrapper expects the model to return a tuple `(probs, sigma)`.
    - `probs` should have shape `(batch_size, num_classes)`.
    - `sigma` should have shape `(batch_size,)`.
    - The regularisation term penalises large uncertainty values.
    - This formulation follows the idea proposed in :footcite:t:`araujo2020dr`.

    Example
    -------
    >>> base_loss = nn.CrossEntropyLoss()
    >>> loss_wrapper = GaussianUncertaintyLossWrapper(base_loss, alpha=0.5)
    >>> probs = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    >>> sigma = torch.tensor([0.5, 0.3])
    >>> y_true = torch.tensor([0, 1])
    >>> loss = loss_wrapper((probs, sigma), y_true)
    >>> print(loss)
    """

    def __init__(
        self,
        base_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        alpha: float = 0.5,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.alpha = alpha

        if not callable(base_loss):
            raise ValueError("base_loss must be a callable function or object.")

    def forward(
        self, y_pred: tuple[torch.Tensor, torch.Tensor], y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the total loss.

        Parameters
        ----------
        y_pred : tuple[torch.Tensor, torch.Tensor]
            Tuple containing:
            - probs: predicted class probabilities,
              shape `(batch_size, num_classes)`
            - sigma: predicted standard deviation,
              shape `(batch_size,)`

        y_true : torch.Tensor
            Ground-truth labels or targets.
            Shape depends on the chosen base loss.

        Returns
        -------
        torch.Tensor
            Scalar loss value combining the base loss and the uncertainty penalty.
        """

        probs, sigma = y_pred
        base_loss_value = self.base_loss(probs, y_true)
        sigma_penalty = (1 - self.alpha) * torch.mean(torch.pow(sigma, 2))
        total_loss = base_loss_value + sigma_penalty
        return total_loss
