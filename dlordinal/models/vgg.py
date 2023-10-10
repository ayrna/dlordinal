from itertools import chain
from typing import Callable, List, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models.vgg as vgg
from torch import nn

from ..layers import VGGOrdinalFullyConnected, activation_function_by_name
from .experiment_model import ExperimentModel


class VGGOrdinalECOC(ExperimentModel):
    """
    VGG model with ordinal fully connected layer.
    """

    classifier: VGGOrdinalFullyConnected
    target_class: torch.Tensor

    def __init__(
        self,
        features: nn.Module,
        num_classes: int,
        activation_function: Callable[[], nn.Module],
        init_weights: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        features : nn.Module
            Features module.
        num_classes : int
            Number of classes.
        activation_function : Callable[[], nn.Module]
            Activation function.
        init_weights : bool, default=True
            Initialize weights.
        """

        super(VGGOrdinalECOC, self).__init__()

        self.features = nn.Sequential(
            features, nn.AdaptiveAvgPool2d((7, 7)), nn.Flatten(start_dim=1)
        )
        self.classifier = VGGOrdinalFullyConnected(
            input_size=512 * 7 * 7,
            activation_function=activation_function,
            num_classes=num_classes,
        )
        if init_weights:
            self._initialize_weights()

        # Reference vectors for each class, for predictions
        target_class = np.ones((num_classes, num_classes - 1), dtype=np.float32)
        target_class[np.triu_indices(num_classes, 0, num_classes - 1)] = 0.0
        self.target_class = torch.tensor(target_class).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def regularized_parameters(self) -> List[nn.parameter.Parameter]:
        """
        Get the regularized parameters.
        """
        return list(
            chain.from_iterable(
                chain(
                    cf[0].parameters(), cf[3].parameters()  # type: ignore
                )  # type: ignore
                for cf in self.classifier.classifiers
            )
        )

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward(x)
        return -torch.cdist(x, self.target_class.to(x.device))

    def on_batch_end(self):
        pass


def make_layers(
    cfg: List[Union[str, int]],
    activation_function: Callable[[], nn.Module],
    batch_norm: bool = False,
) -> nn.Sequential:
    """
    Make layers for the VGG model.

    Parameters
    ----------
    cfg : List[Union[str, int]]
        List of layers.
    activation_function : Callable[[], nn.Module]
        Activation function.
    batch_norm : bool, default=False
        Use batch normalization.
    """
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), activation_function()]
            else:
                layers += [conv2d, activation_function()]
            in_channels = v
    return nn.Sequential(*layers)


def vgg11_ecoc(
    activation_function: Union[str, Callable[[], nn.Module]],
    batch_norm: bool = False,
    **kwargs
):
    """
    VGG 11-layer model (configuration "A") with ordinal fully connected layer.

    Parameters
    ----------
    activation_function : Union[str, Callable[[], nn.Module]]
        Activation function.
    batch_norm : bool, default=False
        Use batch normalization.
    """
    if isinstance(activation_function, str):
        activation_function = activation_function_by_name[activation_function]
    return VGGOrdinalECOC(
        make_layers(vgg.cfgs["A"], activation_function, batch_norm),
        activation_function=activation_function,
        **kwargs
    )


def vgg13_ecoc(
    activation_function: Union[str, Callable[[], nn.Module]],
    batch_norm: bool = False,
    **kwargs
):
    """
    VGG 13-layer model (configuration "B") with ordinal fully connected layer.

    Parameters
    ----------
    activation_function : Union[str, Callable[[], nn.Module]]
        Activation function.
    batch_norm : bool, default=False
        Use batch normalization.
    """
    if isinstance(activation_function, str):
        activation_function = activation_function_by_name[activation_function]
    return VGGOrdinalECOC(
        make_layers(vgg.cfgs["B"], activation_function, batch_norm),
        activation_function=activation_function,
        **kwargs
    )


def vgg16_ecoc(
    activation_function: Union[str, Callable[[], nn.Module]],
    batch_norm: bool = False,
    **kwargs
):
    """
    VGG 16-layer model (configuration "D") with ordinal fully connected layer.

    Parameters
    ----------
    activation_function : Union[str, Callable[[], nn.Module]]
        Activation function.
    batch_norm : bool, default=False
        Use batch normalization.
    """
    if isinstance(activation_function, str):
        activation_function = activation_function_by_name[activation_function]
    return VGGOrdinalECOC(
        make_layers(vgg.cfgs["D"], activation_function, batch_norm),
        activation_function=activation_function,
        **kwargs
    )


def vgg19_ecoc(
    activation_function: Union[str, Callable[[], nn.Module]],
    batch_norm: bool = False,
    **kwargs
):
    """
    VGG 19-layer model (configuration "E") with ordinal fully connected layer.

    Parameters
    ----------
    activation_function : Union[str, Callable[[], nn.Module]]
        Activation function.
    batch_norm : bool, default=False
        Use batch normalization.
    """
    if isinstance(activation_function, str):
        activation_function = activation_function_by_name[activation_function]
    return VGGOrdinalECOC(
        make_layers(vgg.cfgs["E"], activation_function, batch_norm),
        activation_function=activation_function,
        **kwargs
    )
