import pytest
import torch
from torch import nn

from dlordinal.losses import GaussianUncertaintyLossWrapper


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
    params=[0.0, 0.5, 1.0],
    ids=["alpha_0", "alpha_0.5", "alpha_1"],
)
def alpha(request):
    return request.param


@pytest.fixture
def base_config(alpha):
    return {
        "alpha": alpha,
    }


@pytest.fixture
def batch_data():
    batch_size = 6
    num_classes = 4

    log_probs = torch.log(torch.softmax(torch.randn(batch_size, num_classes), dim=-1))
    sigma = torch.rand(batch_size)
    y_true = torch.randint(0, num_classes, (batch_size,))

    return log_probs, sigma, y_true


@pytest.fixture
def batch_data_on_device(batch_data, torch_device):
    log_probs, sigma, y_true = batch_data

    log_probs = log_probs.to(torch_device).requires_grad_(True)
    sigma = sigma.to(torch_device).requires_grad_(True)
    y_true = y_true.to(torch_device)

    return log_probs, sigma, y_true


def base_loss(log_probs, y):
    return -log_probs[torch.arange(log_probs.size(0)), y].mean()


def test_initialization(base_config):
    loss = GaussianUncertaintyLossWrapper(base_loss, **base_config)

    assert isinstance(loss, nn.Module)
    assert loss.alpha == base_config["alpha"]
    assert loss.base_loss is base_loss


def test_forward_pass(base_config, batch_data_on_device):
    loss_fn = GaussianUncertaintyLossWrapper(base_loss, **base_config)

    log_probs, sigma, y_true = batch_data_on_device

    loss = loss_fn((log_probs, sigma), y_true)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    assert torch.isfinite(loss)


def test_backward_pass(base_config, batch_data_on_device):
    loss_fn = GaussianUncertaintyLossWrapper(base_loss, **base_config)

    log_probs, sigma, y_true = batch_data_on_device

    loss = loss_fn((log_probs, sigma), y_true)
    loss.backward()

    assert log_probs.grad is not None
    assert sigma.grad is not None


def test_sigma_penalty_effect(base_config, batch_data_on_device):
    loss_fn = GaussianUncertaintyLossWrapper(base_loss, alpha=0.0)
    loss_fn_no_reg = GaussianUncertaintyLossWrapper(base_loss, alpha=1.0)

    log_probs, sigma, y_true = batch_data_on_device

    loss_with_penalty = loss_fn((log_probs, sigma), y_true)
    loss_no_penalty = loss_fn_no_reg((log_probs, sigma), y_true)
    loss_base = base_loss(log_probs, y_true)

    # alpha=1 => no sigma penalty
    assert loss_no_penalty <= loss_with_penalty + 1e-6

    # loss should be >= base loss due to penalty
    assert loss_with_penalty >= loss_base - 1e-6

    # loss_no_penalty should be equal to base loss
    assert torch.allclose(loss_no_penalty, loss_base, atol=1e-6)


def test_sigma_penalty_is_positive(base_config, batch_data_on_device):
    loss_fn = GaussianUncertaintyLossWrapper(base_loss, **base_config)

    log_probs, sigma, y_true = batch_data_on_device

    loss_with_penalty = loss_fn((log_probs, sigma), y_true)
    loss_base = base_loss(log_probs, y_true)

    # isolate penalty by subtracting base loss
    penalty = loss_with_penalty - loss_base

    assert torch.allclose(penalty, (1 - loss_fn.alpha) * sigma.pow(2).mean(), atol=1e-6)


def test_deterministic_forward(base_config, batch_data_on_device):
    loss_fn = GaussianUncertaintyLossWrapper(base_loss, **base_config)

    log_probs, sigma, y_true = batch_data_on_device

    out1 = loss_fn((log_probs, sigma), y_true)
    out2 = loss_fn((log_probs, sigma), y_true)

    assert torch.allclose(out1, out2)


def test_no_nan_or_inf(base_config, batch_data_on_device):
    loss_fn = GaussianUncertaintyLossWrapper(base_loss, **base_config)

    log_probs, sigma, y_true = batch_data_on_device

    loss = loss_fn((log_probs, sigma), y_true)

    assert torch.isfinite(loss)


def test_loss_increases_with_sigma(base_config, batch_data_on_device):
    loss_fn = GaussianUncertaintyLossWrapper(base_loss, **base_config)

    log_probs, sigma, y_true = batch_data_on_device

    loss1 = loss_fn((log_probs, sigma), y_true)
    loss2 = loss_fn((log_probs, sigma * 2), y_true)

    if loss_fn.alpha == 1.0:
        assert torch.isclose(loss1, loss2)
    else:
        assert loss2 > loss1
