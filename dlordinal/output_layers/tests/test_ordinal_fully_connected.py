import torch
from torch import nn

from dlordinal.output_layers import (
    ResNetOrdinalFullyConnected,
    VGGOrdinalFullyConnected,
)


def test_ordinal_resnet_fc_creation():
    input_size = 10
    num_classes = 5
    resnet_fc = ResNetOrdinalFullyConnected(input_size, num_classes)

    assert isinstance(resnet_fc, ResNetOrdinalFullyConnected)


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


def test_initialisation_VGG():
    input_size = 512
    num_classes = 5
    activation_function = nn.ReLU

    model = VGGOrdinalFullyConnected(input_size, num_classes, activation_function)

    assert len(model.classifiers) == num_classes - 1
    for classifier in model.classifiers:
        assert isinstance(classifier, nn.Sequential)
        layers = list(classifier)
        assert isinstance(layers[0], nn.Linear)
        assert isinstance(layers[1], activation_function)
        assert isinstance(layers[2], nn.Dropout)
        assert isinstance(layers[3], nn.Linear)
        assert isinstance(layers[4], activation_function)
        assert isinstance(layers[5], nn.Dropout)
        assert isinstance(layers[6], nn.Linear)


def test_forward_VGG():
    input_size = 512
    num_classes = 5
    activation_function = nn.ReLU

    model = VGGOrdinalFullyConnected(input_size, num_classes, activation_function)
    x = torch.randn(10, input_size)  # Batch size of 10
    output = model(x)

    assert output.shape == (10, num_classes - 1)
    assert torch.all(output >= 0) and torch.all(output <= 1)
