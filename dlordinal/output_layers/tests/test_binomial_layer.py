import pytest
import torch
from torch import nn

from dlordinal.output_layers import BinomialLayer


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ]
)
def torch_device(request):
    """Fixture to parametrize tests across 'cpu' and 'cuda' (if available)."""
    return torch.device(request.param)


@pytest.fixture(
    params=[4, 7],
    ids=["4_classes", "7_classes"],
)
def num_classes(request):
    return request.param


@pytest.fixture
def base_config(num_classes):
    """Basic configuration to instantiate the loss."""
    in_features = 10
    return {
        "in_features": in_features,
        "num_classes": num_classes,
    }


@pytest.fixture
def input_data(base_config):
    """Generates a batch of inputs and targets (default to CPU)."""
    batch_size = 4
    num_classes = base_config["num_classes"]
    in_features = base_config["in_features"]
    # Random inputs (features)
    inputs = torch.randn(batch_size, in_features, requires_grad=True)
    # Random targets
    targets = torch.randint(0, num_classes, (batch_size,))
    return inputs, targets


@pytest.fixture
def input_data_on_device(input_data, torch_device):
    """Moves input data to the specified device."""
    inputs, targets = input_data
    inputs = inputs.to(torch_device).detach().requires_grad_(True)
    targets = targets.to(torch_device)
    return inputs, targets


def test_initialization(base_config, torch_device):
    """Tests that the class is instantiated correctly."""
    layer = BinomialLayer(**base_config).to(torch_device)
    assert isinstance(layer, nn.Module)
    assert layer.num_classes == base_config["num_classes"]
    assert layer.p_layer.in_features == base_config["in_features"]
    assert layer.p_layer.out_features == 1


def test_doc_example():
    layer = BinomialLayer(in_features=5, num_classes=3)
    input = torch.randn(2, 5)
    probs = layer(input)
    print(probs)


def test_forward_pass(base_config, input_data_on_device, torch_device):
    """Tests that the forward pass produces valid probabilities."""
    layer = BinomialLayer(**base_config).to(torch_device)
    inputs, _ = input_data_on_device

    probs = layer(inputs)
    assert probs.shape == (inputs.shape[0], base_config["num_classes"])
    assert torch.all(probs >= 0) and torch.all(probs <= 1)
    assert torch.allclose(
        probs.sum(dim=1), torch.ones(inputs.shape[0], device=torch_device)
    )


def test_backward_pass(base_config, input_data_on_device, torch_device):
    """Tests that the backward pass computes gradients without errors."""
    layer = BinomialLayer(**base_config).to(torch_device)
    inputs, targets = input_data_on_device

    probs = layer(inputs)
    loss = -torch.log(probs[torch.arange(inputs.shape[0]), targets]).mean()
    loss.backward()

    assert inputs.grad is not None
    assert layer.p_layer.weight.grad is not None
    assert layer.p_layer.bias.grad is not None


def is_unimodal(p: torch.Tensor) -> bool:
    """
    Checks discrete unimodality for a probability vector.
    """
    p = p.detach().float()

    mode = torch.argmax(p)

    if mode > 0:
        left = p[: mode + 1]
        if not torch.all(left[1:] >= left[:-1] - 1e-6):
            return False

    if mode < len(p) - 1:
        right = p[mode:]
        if not torch.all(right[1:] <= right[:-1] + 1e-6):
            return False

    return True


def test_output_is_unimodal(base_config, input_data_on_device, torch_device):
    layer = BinomialLayer(**base_config).to(torch_device)
    inputs, _ = input_data_on_device
    probs = layer(inputs)

    for i in range(probs.shape[0]):
        assert is_unimodal(probs[i]), f"Not unimodal at sample {i}"


def test_deterministic_forward(base_config, input_data_on_device, torch_device):
    layer = BinomialLayer(**base_config).to(torch_device)
    inputs, _ = input_data_on_device

    out1 = layer(inputs)
    out2 = layer(inputs)

    assert torch.allclose(out1, out2)


def test_deterministic_init(base_config, torch_device):
    torch.manual_seed(42)
    layer1 = BinomialLayer(**base_config).to(torch_device)

    torch.manual_seed(42)
    layer2 = BinomialLayer(**base_config).to(torch_device)

    for p1, p2 in zip(layer1.parameters(), layer2.parameters()):
        assert torch.allclose(p1, p2)
