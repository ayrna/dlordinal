import torch

from ..stick_breaking_layer import StickBreakingLayer


def test_stick_breaking_layer_creation():
    input_shape = 5
    num_classes = 3
    layer = StickBreakingLayer(input_shape, num_classes)

    # Check that the layer was created correctly
    assert isinstance(layer, StickBreakingLayer)


def test_stick_breaking_layer():
    input_shape = 5
    num_classes = 3
    layer = StickBreakingLayer(input_shape, num_classes)

    input_data = torch.randn(10, input_shape)

    # Compute logits from stick breaking layer
    logits = layer(input_data)

    print(logits)

    # Check that logits have the expected shape
    assert logits.shape == (10, num_classes)

    # Check logit values
    assert torch.isfinite(logits).all()


if __name__ == "__main__":
    test_stick_breaking_layer_creation()
    test_stick_breaking_layer()
