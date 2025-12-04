import numpy as np
import pytest
import torch
import torch.nn as nn

from dlordinal.losses.sord import SORDLoss, create_prox_mat

# --- FIXTURES ---


@pytest.fixture
def base_config():
    """Basic configuration to instantiate the loss."""
    num_classes = 5
    # Must have targets to calculate prox matrix in certain modes
    dummy_train_targets = torch.randint(0, num_classes, (100,))
    return {
        "alpha": 1.0,
        "num_classes": num_classes,
        "train_targets": dummy_train_targets,
        "use_logits": True,
        "prox": False,
        "ftype": "max",
    }


@pytest.fixture
def input_data(base_config):
    """Generates a batch of inputs (logits) and targets."""
    batch_size = 4
    num_classes = base_config["num_classes"]
    # Random inputs (logits)
    inputs = torch.randn(batch_size, num_classes, requires_grad=True)
    # Random targets
    targets = torch.randint(0, num_classes, (batch_size,))
    return inputs, targets


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
    params=[
        # 1. Standard Ordinal Distance Mode (prox=False) - Ftype is irrelevant here
        ("max", False),
        # 2. Proximity Matrix Modes (prox=True) - All 6 Ftypes
        ("max", True),
        ("norm_max", True),
        ("norm_log", True),
        ("log", True),
        ("norm_division", True),
        ("division", True),
    ]
)
def prox_ftype_config(request, base_config):
    """Parametrized fixture for all 7 logical prox/ftype combinations in SORDLoss."""
    ftype, prox = request.param
    config = base_config.copy()
    config["ftype"] = ftype
    config["prox"] = prox
    return config


# --- TESTS ---


def test_initialization_and_forward(prox_ftype_config, input_data, torch_device):
    """
    Tests basic instantiation and forward pass across all prox/ftype configurations.
    """
    config = prox_ftype_config

    loss_fn = SORDLoss(**config).to(torch_device)
    inputs, targets = input_data
    inputs_dev = inputs.to(torch_device)
    targets_dev = targets.to(torch_device)

    loss = loss_fn(inputs_dev, targets_dev)

    assert isinstance(loss_fn, nn.Module)
    assert torch.is_tensor(loss)
    assert loss.dim() == 0
    assert not torch.isnan(loss)

    # Check if buffers were created in PROX mode
    if config["prox"]:
        assert hasattr(loss_fn, "prox_mat")
        assert loss_fn.prox_mat.device.type == torch_device.type


def test_gradient_propagation(prox_ftype_config, input_data, torch_device):
    """Verifies that gradients flow backwards for all configurations."""
    config = prox_ftype_config
    loss_fn = SORDLoss(**config).to(torch_device)
    inputs, targets = input_data
    inputs_dev = inputs.to(torch_device).detach().requires_grad_(True)
    targets_dev = targets.to(torch_device)

    loss = loss_fn(inputs_dev, targets_dev)
    loss.backward()

    assert inputs_dev.grad is not None
    # Check if the sum of absolute gradients is greater than zero
    assert torch.sum(torch.abs(inputs_dev.grad)) > 1e-6
    assert inputs_dev.grad.device.type == inputs_dev.device.type


def test_create_prox_mat_format():
    """
    Verifies that create_prox_mat returns a 2D Tensor with the correct
    shape and dtype.
    """
    # Simple class counts (e.g., 3 classes)
    dist_dict = {0: 10, 1: 20, 2: 10}

    prox_mat = create_prox_mat(dist_dict, inv=False)

    assert torch.is_tensor(prox_mat)
    assert prox_mat.dim() == 2
    assert prox_mat.shape == (3, 3)
    # The function returns a Tensor, so float should be the default
    # PyTorch type (float32/float64)
    assert prox_mat.dtype == torch.float64


def test_create_prox_mat_numerical_check(torch_device):
    """
    Verifies the exact numerical output for a known, simple input (inv=False).
    Input: {0: 10, 1: 10, 2: 10}. Denominator=30.
    """
    dist_dict = {0: 10, 1: 10, 2: 10}

    # Expected Matrix (M): M[label1][label2] = -log(numerator / 30)
    # Diagonal M[0,0] = M[1,1] = M[2,2]: -log( (10/2) / 30) = 1.79176
    # Adjacent M[0,1] = M[1,0] = M[1,2] = M[2,1]: -log( (10/2 + 10) / 30) = 0.69315
    # Far M[0,2] = M[2,0]: -log( (10/2 + 10 + 10) / 30) = 0.18232

    expected_mat_np = np.array(
        [
            [1.791759, 0.693147, 0.182322],
            [0.693147, 1.791759, 0.693147],
            [0.182322, 0.693147, 1.791759],
        ]
    )
    expected_mat = torch.from_numpy(expected_mat_np).to(torch_device)

    # Test inv=False
    prox_mat_false = create_prox_mat(dist_dict, inv=False).to(torch_device)
    assert torch.allclose(prox_mat_false, expected_mat, atol=1e-5)

    # Test inv=True (Prox_mat_inv = 1 / Prox_mat_false)
    prox_mat_true = create_prox_mat(dist_dict, inv=True).to(torch_device)
    expected_mat_inv = 1 / expected_mat
    assert torch.allclose(prox_mat_true, expected_mat_inv, atol=1e-5)


def test_relative_sensitivity_deterioration(
    prox_ftype_config, input_data, torch_device
):
    """
    Verifies that increasing the logit of an incorrect class far from the target
    increases the loss across all configurations (Loss(B) > Loss(A)).
    """
    config = prox_ftype_config
    inputs, targets = input_data
    device = torch_device

    input_A = inputs[0:1].to(device)
    target_A = targets[0:1].to(device)
    target_idx = target_A.item()
    num_classes = config["num_classes"]

    # Find the index farthest from the target (corner cases for ordinal loss)
    # If target is near 0, the farthest is N-1. If target is near N-1, the farthest
    # is 0.
    if target_idx < num_classes / 2:
        incorrect_idx = num_classes - 1
    else:
        incorrect_idx = 0

    # Ensure the incorrect index is not the target index
    if incorrect_idx == target_idx:
        incorrect_idx = (incorrect_idx + 1) % num_classes

    loss_fn = SORDLoss(**config).to(device)

    # Condition A: Base loss
    loss_A = loss_fn(input_A, target_A)

    # Condition B: Worse inputs. Artificially boost the logit
    # of the incorrect, far-away class.
    input_B = input_A.clone()
    input_B[0, incorrect_idx] += 10.0  # Large confidence boost towards error

    loss_B = loss_fn(input_B, target_A)

    # Loss B must be greater than Loss A
    assert loss_B.item() > loss_A.item(), (
        f"Loss did not increase when prediction worsened. Config: {config}."
        f" Loss A: {loss_A.item()}, Loss B: {loss_B.item()}"
    )


def test_relative_sensitivity_improvement(prox_ftype_config, torch_device):
    """
    Verifies that increasing the logit of the target class reduces the loss
    across all configurations (Loss(B) < Loss(A)).
    """
    config = prox_ftype_config
    device = torch_device

    targets = torch.tensor([2, 3, 1]).to(device)
    inputs_A = torch.tensor(
        [
            [1, 2, 3, 4, 0],
            [0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0],
        ],
        requires_grad=True,
        dtype=torch.float32,
    ).to(device)

    inputs_B = torch.tensor(
        [
            [1, 2, 4, 3, 0],
            [0, 1, 2, 4, 3],
            [3, 4, 2, 1, 0],
        ],
        requires_grad=True,
        dtype=torch.float32,
    ).to(device)

    loss_fn = SORDLoss(**config).to(device)

    loss_A = loss_fn(inputs_A, targets)
    loss_B = loss_fn(inputs_B, targets)

    # The loss with better confidence (B) must be strictly smaller than the base loss (A).
    assert loss_B.item() < loss_A.item(), (
        f"Loss did not decrease when prediction improved. Config: {config}."
        f" Loss A: {loss_A.item()}, Loss B: {loss_B.item()}"
    )


def test_exact_value_noprox(torch_device):
    inputs = torch.tensor([[1.5, 2.5, 5.0, 2.0]], device=torch_device)
    targets = torch.tensor([2], device=torch_device)

    loss_fn = SORDLoss(
        alpha=1.0,
        num_classes=4,
        use_logits=True,
        prox=False,
        train_targets=None,
    ).to(torch_device)

    loss = loss_fn(inputs, targets)
    expected_loss = 1.48472126591998

    assert torch.isclose(
        loss,
        torch.tensor(expected_loss, dtype=loss.dtype, device=torch_device),
        atol=1e-5,
    ), f"Exact loss value mismatch. Expected: {expected_loss}, Got: {loss.item()}"
