from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module


class COPOC(Module):
    """Implements the  Conformal Predictions for OC (COPOC) output layer(s) from :footcite:t:`dey2023conformal`,
    which enforce unimodality in the output probabilities in a non-parametric way."""

    def __init__(
        self,
        phi: Callable[[Tensor], Tensor] = lambda x: torch.abs(x),
        psi: Callable[[Tensor], Tensor] = lambda x: -torch.abs(x),
    ) -> None:
        """

        Parameters
        ----------
        phi: Callable[[Tensor], Tensor]
            Non-negative transformation function. Default is absolute value function.
        psi: Callable[[Tensor], Tensor]
            Strictly monotonic decreasing bijective function. Default is negative absolute value function.
        """
        super().__init__()
        self.phi = phi
        self.psi = psi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, num_classes).

        Returns
        -------
        probs : torch.Tensor
            Unimodal output probabilities of shape (batch_size, num_classes).
        """
        v = x.clone()
        v_rest = self.phi(v[:, 1:])
        v = torch.cat([v[:, :1], v_rest], dim=1)

        r = torch.cumsum(v, dim=1)

        z = self.psi(r)

        probs = torch.nn.functional.softmax(z, dim=1)
        return probs
