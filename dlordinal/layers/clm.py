import warnings
from math import sqrt

import torch
from torch.nn import Module


class CLM(Module):
    """
    Implementation of the cumulative link model from :footcite:t:`vargas2020clm` as a torch layer.
    Different link functions can be used, including logit, probit and cloglog.

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

    Example
    ---------
    >>> import torch
    >>> from dlordinal.layers import CLM

    >>> inp = torch.randn(10, 5)
    >>> fc = torch.nn.Linear(5, 1)
    >>> clm = CLM(5, "logit")

    >>> output = clm(fc(inp))
    >>> print(output)
    tensor([[0.4704, 0.0063, 0.0441, 0.0423, 0.4369],
        [0.2496, 0.0048, 0.0349, 0.0363, 0.6745],
        [0.6384, 0.0058, 0.0393, 0.0357, 0.2808],
        [0.4862, 0.0063, 0.0441, 0.0420, 0.4214],
        [0.3768, 0.0060, 0.0425, 0.0421, 0.5327],
        [0.4740, 0.0063, 0.0441, 0.0422, 0.4334],
        [0.2868, 0.0052, 0.0378, 0.0387, 0.6315],
        [0.2583, 0.0049, 0.0356, 0.0369, 0.6643],
        [0.1811, 0.0038, 0.0281, 0.0300, 0.7570],
        [0.5734, 0.0062, 0.0423, 0.0392, 0.3389]], grad_fn=<CopySlices>)
    """

    def __init__(
        self, num_classes, link_function, min_distance=0.0, clip_warning=True, **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.link_function = link_function
        self.min_distance = min_distance
        self.clip_warning = clip_warning
        self.dist = torch.distributions.Normal(0, 1)

        self.thresholds_b = torch.nn.Parameter(data=torch.Tensor(1), requires_grad=True)
        torch.nn.init.uniform_(self.thresholds_b, 0.0, 0.1)

        self.thresholds_a = torch.nn.Parameter(
            data=torch.Tensor(self.num_classes - 2), requires_grad=True
        )
        torch.nn.init.uniform_(
            self.thresholds_a,
            sqrt((1.0 / (self.num_classes - 2)) / 2),
            sqrt(1.0 / (self.num_classes - 2)),
        )

        self.clip_warning_shown = False

    def _convert_thresholds(self, b, a, min_distance):
        a = a**2
        a = a + min_distance
        thresholds_param = torch.cat((b, a), dim=0)
        th = torch.sum(
            torch.tril(
                torch.ones(
                    (self.num_classes - 1, self.num_classes - 1), device=a.device
                ),
                diagonal=-1,
            )
            * torch.reshape(
                torch.tile(thresholds_param, (self.num_classes - 1,)).to(a.device),
                shape=(self.num_classes - 1, self.num_classes - 1),
            ),
            dim=(1,),
        )
        return th

    def _clm(self, projected: torch.Tensor, thresholds: torch.Tensor):
        projected = torch.reshape(projected, shape=(-1,))

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
            if self.clip_warning and not self.clip_warning_shown:
                warnings.warn(
                    "The output value of the CLM layer is out of the range [-10, 10]."
                    " Clipping value prior to applying the link function for numerical"
                    " stability."
                )
            z3 = torch.clip(a - b, -10, 10)
            self.clip_warning_shown = True

        if self.link_function == "probit":
            a3T = self.dist.cdf(z3)
        elif self.link_function == "cloglog":
            a3T = 1 - torch.exp(-torch.exp(z3))
        else:
            a3T = 1.0 / (1.0 + torch.exp(-z3))

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
            self.thresholds_b, self.thresholds_a, self.min_distance
        )

        return self._clm(x, thresholds)
