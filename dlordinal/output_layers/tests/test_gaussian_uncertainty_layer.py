import pytest
import torch
from torch import nn

from dlordinal.output_layers import GaussianUncertaintyLayer


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
    return torch.device(request.param)


@pytest.fixture(
    params=[4, 7],
    ids=["4_classes", "7_classes"],
)
def num_classes(request):
    return request.param


@pytest.fixture
def base_config(num_classes):
    in_features = 10
    return {
        "in_features": in_features,
        "num_classes": num_classes,
    }


@pytest.fixture
def input_data(base_config):
    batch_size = 4
    num_classes = base_config["num_classes"]
    in_features = base_config["in_features"]

    inputs = torch.randn(batch_size, in_features, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size,))
    return inputs, targets


@pytest.fixture
def input_data_on_device(input_data, torch_device):
    inputs, targets = input_data
    inputs = inputs.to(torch_device).detach().requires_grad_(True)
    targets = targets.to(torch_device)
    return inputs, targets


def test_initialization(base_config, torch_device):
    layer = GaussianUncertaintyLayer(**base_config).to(torch_device)

    assert isinstance(layer, nn.Module)
    assert layer.num_classes == base_config["num_classes"]

    assert layer.mu_layer.in_features == base_config["in_features"]
    assert layer.mu_layer.out_features == 1

    assert layer.sigma_layer.in_features == base_config["in_features"]
    assert layer.sigma_layer.out_features == 1


def test_doc_example():
    layer = GaussianUncertaintyLayer(in_features=5, num_classes=3)
    input = torch.randn(2, 5)
    probs, sigma = layer(input)
    print(probs)


def test_forward_pass(base_config, input_data_on_device, torch_device):
    layer = GaussianUncertaintyLayer(**base_config).to(torch_device)
    inputs, _ = input_data_on_device

    probs, sigma = layer(inputs)

    # Shape checks
    assert probs.shape == (inputs.shape[0], base_config["num_classes"])
    assert sigma.shape == (inputs.shape[0],)

    # Probability constraints
    assert torch.all(probs >= 0)
    assert torch.all(probs <= 1)

    # Normalization
    assert torch.allclose(
        probs.sum(dim=1),
        torch.ones(inputs.shape[0], device=torch_device),
    )

    # Sigma constraints
    assert torch.all(sigma > 0)


def test_backward_pass(base_config, input_data_on_device, torch_device):
    layer = GaussianUncertaintyLayer(**base_config).to(torch_device)
    inputs, targets = input_data_on_device

    probs, sigma = layer(inputs)

    loss = -torch.log(probs[torch.arange(inputs.shape[0]), targets]).mean()
    loss.backward()

    assert inputs.grad is not None
    assert layer.mu_layer.weight.grad is not None
    assert layer.mu_layer.bias.grad is not None
    assert layer.sigma_layer.weight.grad is not None
    assert layer.sigma_layer.bias.grad is not None


def is_unimodal(p: torch.Tensor) -> bool:
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
    layer = GaussianUncertaintyLayer(**base_config).to(torch_device)
    inputs, _ = input_data_on_device

    probs, _ = layer(inputs)

    for i in range(probs.shape[0]):
        assert is_unimodal(probs[i]), f"Not unimodal at sample {i}"


def test_deterministic_forward(base_config, input_data_on_device, torch_device):
    layer = GaussianUncertaintyLayer(**base_config).to(torch_device)
    inputs, _ = input_data_on_device

    out1 = layer(inputs)
    out2 = layer(inputs)

    probs1, sigma1 = out1
    probs2, sigma2 = out2

    assert torch.allclose(probs1, probs2)
    assert torch.allclose(sigma1, sigma2)


def test_deterministic_init(base_config, torch_device):
    torch.manual_seed(42)
    layer1 = GaussianUncertaintyLayer(**base_config).to(torch_device)

    torch.manual_seed(42)
    layer2 = GaussianUncertaintyLayer(**base_config).to(torch_device)

    for p1, p2 in zip(layer1.parameters(), layer2.parameters()):
        assert torch.allclose(p1, p2)


def test_gaussian_uncertainty_layer_small_sigma_does_not_underflow():
    num_classes = 5
    layer = GaussianUncertaintyLayer(in_features=3, num_classes=num_classes)
    with torch.no_grad():
        layer.mu_layer.weight.zero_()
        layer.mu_layer.bias.fill_(0.5)
        layer.sigma_layer.weight.zero_()
        layer.sigma_layer.bias.fill_(-6.0)  # softplus(-6) ~ 0.0025
    probs, sigma = layer(torch.randn(4, 3))
    assert torch.all(sigma < 0.01)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=1), torch.ones(4))
    # mu = 0.5 lies halfway between classes 0 and 1
    assert torch.allclose(probs[:, 0], torch.full((4,), 0.5), atol=1e-5)
    assert torch.allclose(probs[:, 1], torch.full((4,), 0.5), atol=1e-5)
    loss = torch.nn.functional.nll_loss(torch.log(probs), torch.tensor([0, 1, 0, 1]))
    assert torch.isfinite(loss)


def test_gaussian_uncertainty_layer_matches_normalised_pdf():
    torch.manual_seed(0)
    layer = GaussianUncertaintyLayer(in_features=3, num_classes=6)
    x = torch.randn(8, 3)
    probs, sigma = layer(x)
    mu = layer.mu_layer(x).squeeze(-1)
    k = torch.arange(6, dtype=x.dtype)
    pdf = torch.exp(-((k[None, :] - mu[:, None]) ** 2) / (2 * sigma[:, None] ** 2))
    assert torch.allclose(probs, pdf / pdf.sum(dim=1, keepdim=True), atol=1e-6)
