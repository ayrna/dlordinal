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
    >>> output = torch.nn.functional.softmax(copoc(fc(inp)),dim=1)
    >>> print(output)
    tensor([[0.1898, 0.1901, 0.2568, 0.2196, 0.1436],
            [0.4538, 0.3191, 0.1412, 0.0529, 0.0330],
            [0.3371, 0.2554, 0.2151, 0.1047, 0.0876],
            [0.1859, 0.2073, 0.2658, 0.1889, 0.1520],
            [0.3306, 0.2195, 0.1982, 0.1303, 0.1214],
            [0.2132, 0.3768, 0.1590, 0.1278, 0.1232],
            [0.1531, 0.1544, 0.2094, 0.2451, 0.2381],
            [0.4986, 0.2240, 0.1689, 0.0590, 0.0495],
            [0.5838, 0.2201, 0.1289, 0.0507, 0.0166],
            [0.1639, 0.1969, 0.2100, 0.2347, 0.1946]], grad_fn=<SoftmaxBackward0>)
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
            Logits of the unimodal output layer (batch_size, num_classes).
        """
        # Step 1: Compute η(x) = f(x; θ), which is given by the input tensor.
        n = x.clone()  # η ∈ ℝ^K — raw logits for each class

        # Step 2: Ensure all values are non-negative: v_k = φ(η_k), φ = softplus ensures v_k ≥ 0
        v_rest = self.phi(n[:, 1:])
        v = torch.cat([n[:, :1], v_rest], dim=1)

        # Step 3: Generate cumulative sum: r_k = r_{k-1} + v_k (with r₁ = v₁)
        r = torch.cumsum(v, dim=1)

        # Step 4: Apply symmetric decreasing function: z_k = ψ_E(r_k) = -|r_k|
        z = self.psi(r)  # z ∈ ℝ^K, unimodal due to symmetric log-probability decay

        # Step 5: To turn logits into unimodal probabilities compute class probabilities: p̂_k = softmax(z_k)
        # Here, we only return the logits
        return z
