from typing import Literal

import torch
from torch.nn import Module

from dlordinal.output_layers.utils import stable_sigmoid


class CLM(Module):
    """
    Implementation of the cumulative link models from :footcite:t:`vargas2020clm` as a
    torch layer. Different link functions can be used, including logit, probit
    and cloglog.

    Parameters
    ----------
    num_classes : int
        The number of classes.
    link_function : str
        The link function to use. Can be ``'logit'``, ``'probit'`` or ``'cloglog'``.
    min_distance : float, default=0.0
        The minimum distance between thresholds

    Attributes
    ----------
    num_classes : int
        The number of classes.
    link_function : str
        The link function to use. Can be ``'logit'``, ``'probit'`` or ``'cloglog'``.
    min_distance : float
        The minimum distance between thresholds
    dist_ : torch.distributions.Normal
        The normal (0,1) distribution used to compute the probit link function.
    thresholds_b_ : torch.nn.Parameter
        The torch parameter for the first threshold.
    thresholds_a_ : torch.nn.Parameter
        The torch parameter for the alphas of the thresholds.


    Example
    ---------
    >>> import torch
    >>> from dlordinal.output_layers import CLM
    >>> inp = torch.randn(10, 5)
    >>> fc = torch.nn.Linear(5, 1)
    >>> clm = CLM(5, "logit")
    >>> output = clm(fc(inp))
    >>> print(output)
    tensor([[0.7944, 0.1187, 0.0531, 0.0211, 0.0127],
            [0.4017, 0.2443, 0.1862, 0.0987, 0.0690],
            [0.4619, 0.2381, 0.1638, 0.0814, 0.0548],
            [0.4636, 0.2378, 0.1632, 0.0809, 0.0545],
            [0.4330, 0.2419, 0.1746, 0.0893, 0.0612],
            [0.5006, 0.2309, 0.1495, 0.0716, 0.0473],
            [0.6011, 0.2027, 0.1138, 0.0504, 0.0320],
            [0.5995, 0.2032, 0.1144, 0.0507, 0.0322],
            [0.4014, 0.2443, 0.1863, 0.0988, 0.0691],
            [0.6922, 0.1672, 0.0838, 0.0351, 0.0217]], grad_fn=<CopySlices>)

    """

    def __init__(
        self,
        num_classes: int,
        link_function: Literal["logit", "probit", "cloglog"],
        min_distance: int = 0.0,
        **kwargs,
    ):
        super().__init__()
        if link_function not in ("logit", "probit", "cloglog"):
            raise ValueError(
                f"Invalid link_function '{link_function}'."
                " Expected one of ['logit', 'probit', 'cloglog']."
            )
        self.num_classes = num_classes
        self.link_function = link_function
        self.min_distance = min_distance
        self.dist_ = torch.distributions.Normal(0, 1)

        self.thresholds_b_ = torch.nn.Parameter(
            data=torch.Tensor([0]), requires_grad=True
        )
        self.thresholds_a_ = torch.nn.Parameter(
            data=torch.Tensor([1.0 for _ in range(self.num_classes - 2)]),
            requires_grad=True,
        )

    def _convert_thresholds(self, b, a, min_distance):
        a = a**2
        a = a + min_distance
        thresholds_param = torch.cat((b, a), dim=0)
        th = torch.sum(
            torch.tril(
                torch.ones(
                    (self.num_classes - 1, self.num_classes - 1), device=a.device
                ),
                diagonal=0,
            )
            * torch.reshape(
                torch.tile(thresholds_param, (self.num_classes - 1,)).to(a.device),
                shape=(self.num_classes - 1, self.num_classes - 1),
            ),
            dim=(1,),
        )
        return th

    def _compute_z3(self, projected: torch.Tensor, thresholds: torch.Tensor):
        m = projected.shape[0]
        a = torch.reshape(torch.tile(thresholds, (m,)), shape=(m, -1))
        b = torch.transpose(
            torch.reshape(
                torch.tile(projected, (self.num_classes - 1,)), shape=(-1, m)
            ),
            0,
            1,
        )

        z3 = a - b
        return z3

    def _apply_link_function(self, z3):
        if self.link_function == "probit":
            # Standard normal CDF; avoids the argument validation (and the
            # host-device sync) of torch.distributions.Normal.cdf
            a3T = torch.special.ndtr(z3)
        elif self.link_function == "cloglog":
            # 1 - exp(-exp(z)) cancels to exactly 0 for very negative z; expm1 keeps
            # the small probabilities (and finite log-likelihoods)
            a3T = -torch.expm1(-torch.exp(z3))
        else:  # 'logit'
            a3T = stable_sigmoid(z3)

        return a3T

    def _apply_survival_function(self, z3):
        """Returns ``1 - F(z3)`` computed without cancellation, where ``F`` is the
        inverse link function applied by :meth:`_apply_link_function`."""
        if self.link_function == "probit":
            return torch.special.ndtr(-z3)
        elif self.link_function == "cloglog":
            return torch.exp(-torch.exp(z3))
        else:  # 'logit'
            return torch.sigmoid(-z3)

    def _clm(self, projected: torch.Tensor, thresholds: torch.Tensor):
        projected = torch.reshape(projected, shape=(-1,))

        m = projected.shape[0]
        z3 = self._compute_z3(projected, thresholds)
        cdf = self._apply_link_function(z3)
        sf = self._apply_survival_function(z3)

        zeros = torch.zeros((m, 1), device=projected.device, dtype=cdf.dtype)
        ones = torch.ones((m, 1), device=projected.device, dtype=cdf.dtype)
        cdf = torch.cat((zeros, cdf, ones), dim=1)
        sf = torch.cat((ones, sf, zeros), dim=1)

        # P(y = k) = F(t_k) - F(t_{k-1}) = S(t_{k-1}) - S(t_k). The difference of
        # CDFs cancels to 0 when both are close to 1 (upper tail), so the survival
        # functions are used there instead.
        probs_cdf = cdf[:, 1:] - cdf[:, :-1]
        probs_sf = sf[:, :-1] - sf[:, 1:]
        return torch.where(cdf[:, :-1] > 0.5, probs_sf, probs_cdf)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        output: Tensor
            The output tensor.
        """

        thresholds = self._convert_thresholds(
            self.thresholds_b_, self.thresholds_a_, self.min_distance
        )

        return self._clm(x, thresholds)
