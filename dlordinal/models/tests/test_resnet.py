import pytest
import torch
import torch.nn as nn

from ..resnet import (
    BasicBlock,
    Bottleneck,
    ResNet,
    ResNetOrdinalECOC,
    conv1x1,
    conv3x3,
    resnet18_ecoc,
)


@pytest.fixture
def sample_tensor():
    return torch.randn(1, 3, 224, 224)


def test_conv3x3():
    conv_layer = conv3x3(3, 64)
    assert isinstance(conv_layer, nn.Conv2d)
    input_tensor = torch.randn(1, 3, 224, 224)
    output_tensor = conv_layer(input_tensor)
    assert output_tensor.shape == (1, 64, 224, 224)


def test_conv1x1():
    conv_layer = conv1x1(64, 128)
    assert isinstance(conv_layer, nn.Conv2d)
    input_tensor = torch.randn(1, 64, 224, 224)
    output_tensor = conv_layer(input_tensor)
    assert output_tensor.shape == (1, 128, 224, 224)


def test_BasicBlock():
    basic_block = BasicBlock(
        inplanes=64,
        planes=64,
        activation_function=nn.ReLU,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=nn.BatchNorm2d,
    )

    assert isinstance(basic_block, BasicBlock)

    input_tensor = torch.rand((1, 64, 224, 224))
    output = basic_block(input_tensor)

    assert output.shape == input_tensor.shape


def test_Bottleneck():
    bottleneck_block = Bottleneck(
        inplanes=256,
        planes=64,
        activation_function=nn.ReLU,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=nn.BatchNorm2d,
    )

    assert isinstance(bottleneck_block, Bottleneck)

    input_tensor = torch.rand((1, 256, 224, 224))
    output_tensor = bottleneck_block(input_tensor)

    assert output_tensor.shape == input_tensor.shape


def test_ResNet(sample_tensor):
    resnet = ResNet(
        BasicBlock, [2, 2, 2, 2], activation_function=nn.ReLU, num_classes=1000
    )

    assert isinstance(resnet, ResNet)
    output_tensor = resnet(sample_tensor)
    assert output_tensor.shape[1] == 1000


def test_ResNetOrdinalECOC(sample_tensor):
    resnet_ecoc = ResNetOrdinalECOC(
        BasicBlock, [2, 2, 2, 2], activation_function=nn.ReLU, num_classes=1000
    )

    assert isinstance(resnet_ecoc, ResNetOrdinalECOC)
    output_tensor = resnet_ecoc(sample_tensor)
    assert output_tensor.shape[1] == resnet_ecoc.num_classes - 1


def test_resnet18_ecoc(sample_tensor):
    resnet_ecoc = resnet18_ecoc(activation_function=nn.ReLU, num_classes=1000)
    output_tensor = resnet_ecoc(sample_tensor)
    assert output_tensor.shape[1] == resnet_ecoc.num_classes - 1


if __name__ == "__main__":
    test_conv3x3()
    test_conv1x1()
    test_BasicBlock()
    test_Bottleneck()
    test_ResNet()
    test_ResNetOrdinalECOC()
    test_resnet18_ecoc()
