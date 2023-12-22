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
    """

    def __init__(self, num_classes, link_function, min_distance=0.0, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.link_function = link_function
        self.min_distance = min_distance
        self.dist = torch.distributions.Normal(0, 1)
        self.device = "cpu"

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

    def _convert_thresholds(self, b, a, min_distance):
        a = a**2
        a = a + min_distance
        thresholds_param = torch.cat((b, a), dim=0)
        th = torch.sum(
            torch.tril(
                torch.ones((self.num_classes - 1, self.num_classes - 1)).to(
                    self.device
                ),
                diagonal=-1,
            )
            * torch.reshape(
                torch.tile(thresholds_param, (self.num_classes - 1,)).to(self.device),
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

        if self.link_function == "probit":
            a3T = self.dist.cdf(z3)
        elif self.link_function == "cloglog":
            a3T = 1 - torch.exp(-torch.exp(z3))
        else:
            a3T = 1.0 / (1.0 + torch.exp(-z3))

        ones = torch.ones((m, 1)).to(self.device)
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

    def to(self, device):
        self.device = device

        return self
