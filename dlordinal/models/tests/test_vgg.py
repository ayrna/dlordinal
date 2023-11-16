import pytest
import torch
from torch import nn

# Importa las clases y funciones que deseas probar
from ..vgg import (
    VGGOrdinalECOC,
    make_layers,
    vgg11_ecoc,
    vgg13_ecoc,
    vgg16_ecoc,
    vgg19_ecoc,
)


def test_VGGOrdinalECOC_forward():
    num_classes = 10

    features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1))
    model = VGGOrdinalECOC(
        features=features,
        num_classes=num_classes,
        activation_function=nn.ReLU,
        init_weights=False,
    )

    assert isinstance(model, VGGOrdinalECOC)

    input_tensor = torch.randn((1, 3, 224, 224))

    feature_output = features(input_tensor)
    feature_output_size = (
        feature_output.shape[1] * feature_output.shape[2] * feature_output.shape[3]
    )

    model.classifier = nn.Linear(feature_output_size, num_classes)

    feature_output_flatten = torch.flatten(feature_output, start_dim=1)

    output_tensor = model.classifier(feature_output_flatten)

    expected_output_size = torch.Size([1, num_classes])

    assert output_tensor.shape == pytest.approx(expected_output_size, rel=1e-4)


def test_make_layers():
    layers = make_layers([64, 64, "M", 128, 128, "M"], torch.nn.ReLU)

    input_tensor = torch.randn((1, 3, 224, 224))

    output_tensor = layers(input_tensor)
    assert output_tensor is not None


@pytest.mark.parametrize(
    "vgg_function", [vgg11_ecoc, vgg13_ecoc, vgg16_ecoc, vgg19_ecoc]
)
def test_vgg_ecoc(vgg_function):
    num_classes = 12
    model = vgg_function(activation_function=torch.nn.ReLU, num_classes=num_classes)

    input_tensor = torch.randn((1, 3, 224, 224))

    output_tensor = model.forward(input_tensor)

    assert output_tensor is not None
    assert output_tensor.shape == pytest.approx(
        torch.Size([1, num_classes - 1]), rel=1e-4
    )


if __name__ == "__main__":
    test_VGGOrdinalECOC_forward()
    test_make_layers()
    test_vgg_ecoc()
