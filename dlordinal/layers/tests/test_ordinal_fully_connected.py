import pytest
import torch
import torch.nn as nn

from ..activation_function import activation_function_by_name
from ..ordinal_fully_connected import (
    ResNetOrdinalFullyConnected,
    VGGOrdinalFullyConnected,
)


def test_ordinal_resnet_fc_creation():
    input_size = 10
    num_classes = 5
    resnet_fc = ResNetOrdinalFullyConnected(input_size, num_classes)

    assert isinstance(resnet_fc, ResNetOrdinalFullyConnected)


def test_orindal_VGG_fc_creation():
    input_size = 10
    num_classes = 5
    activation_function = activation_function_by_name["relu"]
    vgg_fc = VGGOrdinalFullyConnected(input_size, num_classes, activation_function)

    assert isinstance(vgg_fc, VGGOrdinalFullyConnected)


@pytest.mark.parametrize("activation_name", ["relu", "elu", "softplus"])
def test_ordinal_vgg_fc_activation_functions(activation_name):
    input_size = 10
    num_classes = 5

    activation_function = activation_function_by_name[activation_name]

    vgg_fc = VGGOrdinalFullyConnected(input_size, num_classes, activation_function)

    input_data = torch.randn(16, input_size)

    output = vgg_fc(input_data)

    # Check that the output has the correct size
    expected_output_size = (16, num_classes - 1)
    assert output.size() == expected_output_size

    # Check that all values are in the range [0, 1] after applying sigmoid
    assert (output >= 0).all()
    assert (output <= 1).all()


def test_ordinal_resnet_fc_output():
    input_size = 10
    num_classes = 5

    resnet_fc = ResNetOrdinalFullyConnected(input_size, num_classes)

    input_data = torch.randn(16, input_size)

    output = resnet_fc(input_data)

    # Check that the output has the correct size
    expected_output_size = (16, num_classes - 1)
    assert output.size() == expected_output_size

    # Check that all values are in the range [0, 1] after applying sigmoid
    assert (output >= 0).all()
    assert (output <= 1).all()


if __name__ == "__main__":
    test_ordinal_resnet_fc_creation()
    test_ordinal_resnet_fc_output()
