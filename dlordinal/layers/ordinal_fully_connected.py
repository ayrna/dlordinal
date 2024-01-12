from typing import Callable

import torch
from torch import nn


class ResNetOrdinalFullyConnected(nn.Module):
    """
    ResNetOrdinalFullyConnected implements the ordinal fully connected layer

    Parameters
    ----------
    input_size: int
        Input size
    num_classes: int
        Number of classes
    """

    classifiers: nn.ModuleList

    def __init__(self, input_size: int, num_classes: int):
        super(ResNetOrdinalFullyConnected, self).__init__()
        self.classifiers = nn.ModuleList(
            [nn.Linear(input_size, 1) for _ in range(num_classes - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        """
        xs = [classifier(x) for classifier in self.classifiers]
        x = torch.cat(xs, dim=1)
        x = torch.sigmoid(x)
        return x


class VGGOrdinalFullyConnected(nn.Module):
    """
    VGGOrdinalFullyConnected implements the ordinal fully connected layer

    Parameters
    ----------
    input_size: int
        Input size
    num_classes: int
        Number of classes
    activation_function: Callable[[], nn.Module]
        Activation function
    """

    classifiers: nn.ModuleList

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        activation_function: Callable[[], nn.Module],
    ):
        super(VGGOrdinalFullyConnected, self).__init__()
        hidden_size = 4096 // (num_classes - 1)
        self.classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    activation_function(),
                    nn.Dropout(),
                    nn.Linear(hidden_size, hidden_size),
                    activation_function(),
                    nn.Dropout(),
                    nn.Linear(hidden_size, 1),
                )
                for _ in range(num_classes - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        """
        xs = [classifier(x) for classifier in self.classifiers]
        x = torch.cat(xs, dim=1)
        x = torch.sigmoid(x)
        return x
