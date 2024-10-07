from typing import Literal

import torch
from torch.nn import Module


class CLMAT(Module):
    def __init__(
        self,
        num_classes: int,
        link_function: Literal["logit", "probit", "cloglog"],
        min_distance: int = 0.0,
        use_weights: bool = False,
        use_gammas: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.link_function = link_function
        self.min_distance = min_distance
        self.use_weights = use_weights
        self.use_gammas = use_gammas
        self.dist_ = torch.distributions.Normal(0, 1)

        if self.use_gammas:
            self.thresholds_gammas_ = torch.nn.Parameter(
                data=torch.Tensor([1.0 for _ in range(self.num_classes - 2)]),
                requires_grad=True,
            )
            self.first_threshold_ = torch.nn.Parameter(
                data=torch.Tensor([0]), requires_grad=True
            )
        else:
            self.thresholds_ = torch.nn.Parameter(
                data=torch.Tensor([float(i) for i in range(self.num_classes - 1)]),
                requires_grad=True,
            )

        if self.use_weights:
            self.weights_ = torch.nn.Parameter(
                data=torch.Tensor([1.0]),
                requires_grad=True,
            )
        else:
            self.register_buffer("weights_", torch.Tensor([1.0]))

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
        return z3

    def _apply_link_function(self, z3):
        if self.link_function == "probit":
            a3T = self.dist_.cdf(z3)
        elif self.link_function == "cloglog":
            a3T = 1 - torch.exp(-torch.exp(z3))
        else:
            a3T = 1.0 / (1.0 + torch.exp(-z3))

        return a3T

    def _clm(
        self, x: torch.Tensor, thresholds: torch.Tensor, weights: torch.Tensor = 1.0
    ):
        projection = weights * x
        projection = torch.reshape(projection, shape=(-1,))

        m = projection.shape[0]
        z3 = self._compute_z3(projection, thresholds)
        a3T = self._apply_link_function(z3)

        ones = torch.ones((m, 1), device=projection.device)
        a3 = torch.cat((a3T, ones), dim=1)
        a3[:, 1:] = a3[:, 1:] - a3[:, 0:-1]

        return a3, weights, x, thresholds

    def forward(self, x):
        if self.use_gammas:
            thresholds = self._convert_thresholds(
                self.first_threshold_, self.thresholds_gammas_, self.min_distance
            )
            return self._clm(x, thresholds, self.weights_)
        else:
            return self._clm(x, self.thresholds_, self.weights_)
