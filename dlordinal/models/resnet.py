from typing import Callable, List, Optional, Type, Union

import numpy as np
import torch
from torch import Tensor, nn

from ..layers import ResNetOrdinalFullyConnected, activation_function_by_name
from .experiment_model import ExperimentModel


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding

    Parameters
    ----------
    in_planes : int
        Number of input planes.
    out_planes : int
        Number of output planes.
    stride : int
        Stride.
    groups : int
        Number of groups.
    dilation : int
        Dilation.
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution

    Parameters
    ----------
    in_planes : int
        Number of input planes.
    out_planes : int
        Number of output planes.
    stride : int
        Stride.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    BasicBlock implements the basic block of ResNet.
    """

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        activation_function: Callable[[], nn.Module],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        Parameters
        ----------
        inplanes : int
            Number of input planes.
        planes : int
            Number of output planes.
        activation_function : Callable[[], nn.Module]
            Activation function.
        stride : int
            Stride.
        downsample : Optional[nn.Module]
            Downsample.
        groups : int
            Number of groups.
        base_width : int
            Base width.
        dilation : int
            Dilation.
        norm_layer : Optional[Callable[..., nn.Module]]
            Normalization layer.
        """

        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.activation = activation_function()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck in torchvision places the stride for downsampling at 3x3
    convolution(self.conv2) while original implementation places the stride at the
    first 1x1 convolution(self.conv1) according to "Deep residual learning for image
    recognition"https://arxiv.org/abs/1512.03385. This variant is also known as
    ResNet V1.5 and improves accuracy according to
    https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    """

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        activation_function: Callable[[], nn.Module],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        Parameters
        ----------
        inplanes : int
            Number of input planes.
        planes : int
            Number of output planes.
        activation_function : Callable[[], nn.Module]
            Activation function.
        stride : int
            Stride.
        downsample : Optional[nn.Module]
            Downsample.
        groups : int
            Number of groups.
        base_width : int
            Base width.
        dilation : int
            Dilation.
        norm_layer : Optional[Callable[..., nn.Module]]
            Normalization layer.
        """

        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.activation = activation_function()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class ResNet(ExperimentModel):
    """
    ResNet implements the ResNet architecture.
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        *,
        activation_function: Union[str, Callable[[], nn.Module]],
        classifier: Callable[[int, int], nn.Module] = nn.Linear,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        Parameters
        ----------
        block : Type[Union[BasicBlock, Bottleneck]]
            Block type.
        layers : List[int]
            List of layers.
        activation_function : Union[str, Callable[[], nn.Module]]
            Activation function.
        classifier : Callable[[int, int], nn.Module]
            Classifier.
        num_classes : int
            Number of classes.
        zero_init_residual : bool
            Zero init residual.
        groups : int
            Number of groups.
        width_per_group : int
            Width per group.
        replace_stride_with_dilation : Optional[List[bool]]
            Replace stride with dilation.
        norm_layer : Optional[Callable[..., nn.Module]]
            Normalization layer.
        """

        super(ResNet, self).__init__()
        if isinstance(activation_function, str):
            activation_function = activation_function_by_name[activation_function]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.activation = activation_function()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], activation_function)
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            activation_function,
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            activation_function,
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            activation_function,
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_classes = num_classes
        self.classifier = classifier(512 * block.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        self.features = nn.Sequential(
            self.conv1,
            self.bn1,
            self.activation,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool,
            nn.Flatten(start_dim=1),
        )

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        activation_function,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                activation_function,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    activation_function,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.features(x)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """
        return self._forward_impl(x)

    def regularized_parameters(self) -> List[nn.parameter.Parameter]:
        """
        Get the regularized parameters.
        """
        return list(self.parameters())

    def on_batch_end(self):
        pass

    def scores(self, x: Tensor) -> Tensor:
        return self.forward(x)


class ResNetOrdinalECOC(ResNet):
    """
    ResNetOrdinalECOC implements the ResNet architecture with ECOC classifier.
    """

    target_class: torch.Tensor

    def __init__(self, *args, **kwargs) -> None:
        """
        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.
        """
        if "classifier" in kwargs:
            raise TypeError("Cannot specify classifier for OBD classifier")
        kwargs["classifier"] = ResNetOrdinalFullyConnected
        super(ResNetOrdinalECOC, self).__init__(*args, **kwargs)

        num_classes = kwargs.get("num_classes", 1000)

        # Reference vectors for each class, for predictions
        target_class = np.ones((num_classes, num_classes - 1), dtype=np.float32)
        target_class[np.triu_indices(num_classes, 0, num_classes - 1)] = 0.0
        self.target_class = torch.tensor(target_class).float()

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """
        x = self.forward(x)
        return -torch.cdist(x, self.target_class.to(x.device))


cfgs = {
    "18": (BasicBlock, [2, 2, 2, 2]),
    "34": (BasicBlock, [3, 4, 6, 3]),
    "50": (Bottleneck, [3, 4, 6, 3]),
    "101": (Bottleneck, [3, 4, 23, 3]),
    "152": (Bottleneck, [3, 8, 36, 3]),
}


def resnet18_ecoc(**kwargs) -> ResNet:
    """
    ResNet18 with ECOC classifier.
    """
    return ResNetOrdinalECOC(*cfgs["18"], **kwargs)


def resnet34_ecoc(**kwargs) -> ResNet:
    """
    ResNet34 with ECOC classifier.
    """
    return ResNetOrdinalECOC(*cfgs["34"], **kwargs)


def resnet50_ecoc(**kwargs) -> ResNet:
    """
    ResNet50 with ECOC classifier.
    """
    return ResNetOrdinalECOC(*cfgs["50"], **kwargs)


def resnet101_ecoc(**kwargs) -> ResNet:
    """
    ResNet101 with ECOC classifier.
    """
    return ResNetOrdinalECOC(*cfgs["101"], **kwargs)


def resnet152_ecoc(**kwargs) -> ResNet:
    """
    ResNet152 with ECOC classifier.
    """
    return ResNetOrdinalECOC(*cfgs["152"], **kwargs)
