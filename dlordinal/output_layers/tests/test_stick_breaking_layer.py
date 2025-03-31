import pytest
import torch

from dlordinal.output_layers import StickBreakingLayer


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_stick_breaking_layer_creation(device):
    input_shape = 5
    num_classes = 3
    layer = StickBreakingLayer(input_shape, num_classes).to(device)

    # Check that the layer was created correctly
    assert isinstance(layer, StickBreakingLayer)


def test_stick_breaking_layer(device):
    input_shape = 5
    num_classes = 3
    layer = StickBreakingLayer(input_shape, num_classes).to(device)

    input_data = torch.randn(10, input_shape).to(device)

    # Compute logits from stick breaking layer
    logits = layer(input_data)

    # Check that logits have the expected shape
    assert logits.shape == (10, num_classes)

    # Check logit values
    assert torch.isfinite(logits).all()
