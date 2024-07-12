import warnings
from typing import Literal

import torch
from torch.nn import Module


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
    clip_warning : bool, default=True
        Whether to print the clipping value warning or not.

    Attributes
    ----------
    num_classes : int
        The number of classes.
    link_function : str
        The link function to use. Can be ``'logit'``, ``'probit'`` or ``'cloglog'``.
    min_distance : float
        The minimum distance between thresholds
    clip_warning : bool
        Whether to print the clipping value warning or not.
    dist_ : torch.distributions.Normal
        The normal (0,1) distribution used to compute the probit link function.
    thresholds_b_ : torch.nn.Parameter
        The torch parameter for the first threshold.
    thresholds_a_ : torch.nn.Parameter
        The torch parameter for the alphas of the thresholds.
    clip_warning_shown_ : bool
        Whether the clipping warning has been shown or not.


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
        clip_warning: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.link_function = link_function
        self.min_distance = min_distance
        self.clip_warning = clip_warning
        self.dist_ = torch.distributions.Normal(0, 1)

        self.thresholds_b_ = torch.nn.Parameter(
            data=torch.Tensor([0]), requires_grad=True
        )
        self.thresholds_a_ = torch.nn.Parameter(
            data=torch.Tensor([1.0 for _ in range(self.num_classes - 2)]),
            requires_grad=True,
        )

        self.clip_warning_shown_ = False

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
        if torch.any(z3 > 10) or torch.any(z3 < -10):
            if self.clip_warning and not self.clip_warning_shown_:
                warnings.warn(
                    f"The output value of the CLM layer (max: {z3.abs().max()}) is out "
                    "of the range [-10, 10]. Clipping value prior to applying the "
                    "link function for numerical stability."
                )
            z3 = torch.clip(a - b, -10, 10)
            self.clip_warning_shown_ = True

        return z3

    def _apply_link_function(self, z3):
        if self.link_function == "probit":
            a3T = self.dist_.cdf(z3)
        elif self.link_function == "cloglog":
            a3T = 1 - torch.exp(-torch.exp(z3))
        else:
            a3T = 1.0 / (1.0 + torch.exp(-z3))

        return a3T

    def _clm(self, projected: torch.Tensor, thresholds: torch.Tensor):
        projected = torch.reshape(projected, shape=(-1,))

        m = projected.shape[0]
        z3 = self._compute_z3(projected, thresholds)
        a3T = self._apply_link_function(z3)

        ones = torch.ones((m, 1), device=projected.device)
        a3 = torch.cat((a3T, ones), dim=1)
        a3[:, 1:] = a3[:, 1:] - a3[:, 0:-1]

        return a3

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
