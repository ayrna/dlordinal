import pytest
import torch

from dlordinal.output_layers import StickBreakingLayer


def test_stick_breaking_layer_creation(device):
    input_shape = 5
    num_classes = 3
    layer = StickBreakingLayer(input_shape, num_classes).to(device)

    # Check that the layer was created correctly
    assert isinstance(layer, StickBreakingLayer)
    # J classes need J - 1 breaks of the stick
    assert layer.fcn1.out_features == num_classes - 1


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


def test_stick_breaking_layer_outputs_normalised_log_probs(device):
    torch.manual_seed(0)
    layer = StickBreakingLayer(8, 6).to(device)
    log_probs = layer(torch.randn(32, 8, device=device))

    assert (log_probs <= 0).all()
    torch.testing.assert_close(
        log_probs.exp().sum(dim=1), torch.ones(32, device=device)
    )
    # Already normalised, so CrossEntropyLoss sees the same log-probabilities
    torch.testing.assert_close(torch.log_softmax(log_probs, dim=1), log_probs)


def test_stick_breaking_layer_matches_reference_formula(device):
    torch.manual_seed(1)
    num_classes = 5
    layer = StickBreakingLayer(4, num_classes).to(device).double()
    x = torch.randn(16, 4, device=device, dtype=torch.float64)

    sigma = torch.sigmoid(layer.fcn1(x))
    expected = torch.empty(16, num_classes, device=device, dtype=torch.float64)
    remaining = torch.ones(16, device=device, dtype=torch.float64)
    for k in range(num_classes - 1):
        expected[:, k] = sigma[:, k] * remaining
        remaining = remaining * (1 - sigma[:, k])
    expected[:, -1] = remaining

    torch.testing.assert_close(layer(x).exp(), expected)
    # Reference values for the first break: p_1 = sigma_1
    torch.testing.assert_close(layer(x).exp()[:, 0], sigma[:, 0])


def test_get_stick_logits_matches_forward(device):
    torch.manual_seed(2)
    layer = StickBreakingLayer(4, 4).to(device)
    x = torch.randn(8, 4, device=device)

    from_breaks = layer.get_stick_logits(torch.sigmoid(layer.fcn1(x)))
    torch.testing.assert_close(from_breaks, layer(x))


@pytest.mark.parametrize("target_class", [0, 3])
def test_stick_breaking_layer_can_be_confident(device, target_class):
    # Probabilities must be able to approach 1 for any class, including the last
    num_classes = 4
    layer = StickBreakingLayer(3, num_classes).to(device)
    bias = torch.full((num_classes - 1,), -30.0, device=device)
    if target_class < num_classes - 1:
        bias[target_class] = 30.0
    with torch.no_grad():
        layer.fcn1.weight.zero_()
        layer.fcn1.bias.copy_(bias)

    probs = layer(torch.randn(2, 3, device=device)).exp()
    assert (probs[:, target_class] > 0.999).all()


def test_stick_breaking_layer_saturated_units_keep_gradients(device):
    layer = StickBreakingLayer(3, 4).to(device)
    with torch.no_grad():
        layer.fcn1.weight.zero_()
        layer.fcn1.bias.copy_(torch.tensor([40.0, -40.0, 40.0], device=device))

    log_probs = layer(torch.randn(5, 3, device=device))
    assert torch.isfinite(log_probs).all()

    # Target the class the saturated first unit rules out
    loss = torch.nn.functional.nll_loss(log_probs, torch.full((5,), 2, device=device))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(layer.fcn1.bias.grad).all()
    assert (layer.fcn1.bias.grad.abs() > 0).all()


def test_stick_breaking_layer_invalid_num_classes():
    with pytest.raises(ValueError, match="num_classes must be at least 2"):
        StickBreakingLayer(3, 1)
