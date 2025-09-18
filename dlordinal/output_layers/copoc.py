from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module


class COPOC(Module):
    """Implements the  Conformal Predictions for OC (COPOC) output layer(s) from :footcite:t:`dey2023conformal`,
    which enforce unimodality in the output probabilities in a non-parametric way.

    Parameters
    ----------
    phi: Callable[[Tensor], Tensor]
        Non-negative transformation function. Default is absolute value function :math:`\\phi(x)=|x|`.
    psi: Callable[[Tensor], Tensor]
        Strictly monotonic decreasing bijective function. Default is negative absolute value function :math:`\\psi(x)=-|x|`.

    Example
    -------
    >>> import torch
    >>> from dlordinal.output_layers import COPOC
    >>> inp = torch.randn(10, 5)
    >>> fc = torch.nn.Linear(5, 5)
    >>> copoc = COPOC()
    >>> output = copoc(fc(inp))
    >>> print(output)
    tensor([[0.2934, 0.2731, 0.1645, 0.1378, 0.1312],
        [0.4051, 0.2438, 0.1590, 0.1261, 0.0660],
        [0.1680, 0.2122, 0.2945, 0.2091, 0.1162],
        [0.1649, 0.2187, 0.2344, 0.1990, 0.1830],
        [0.1225, 0.2699, 0.2895, 0.2231, 0.0951],
        [0.1536, 0.2260, 0.2618, 0.2019, 0.1568],
        [0.3009, 0.2270, 0.1957, 0.1557, 0.1208],
        [0.3659, 0.2500, 0.1863, 0.1504, 0.0474],
        [0.1658, 0.2247, 0.2431, 0.2056, 0.1609],
        [0.5315, 0.2423, 0.1242, 0.0854, 0.0167]], grad_fn=<SoftmaxBackward0>)

    """

    def __init__(
        self,
        phi: Callable[[Tensor], Tensor] = lambda x: torch.abs(x),
        psi: Callable[[Tensor], Tensor] = lambda x: -torch.abs(x),
    ) -> None:
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
