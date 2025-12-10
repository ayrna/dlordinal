import pytest
import torch
import torch.nn.functional as F
from torch import nn

from dlordinal.losses import SLACELoss

# --- FIXTURES ---


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


@pytest.fixture
def base_config():
    """Basic configuration to instantiate the loss."""
    num_classes = 5
    return {
        "alpha": 1.0,
        "num_classes": num_classes,
        "use_logits": True,
    }


@pytest.fixture
def input_data(base_config):
    """Generates a batch of inputs and targets (default to CPU)."""
    batch_size = 4
    num_classes = base_config["num_classes"]
    # Random inputs (logits)
    inputs = torch.randn(batch_size, num_classes, requires_grad=True)
    # Random targets
    targets = torch.randint(0, num_classes, (batch_size,))
    return inputs, targets


@pytest.fixture
def loss_data_on_device(base_config, input_data, torch_device):
    """
    Moves the SLACELoss module and input/target data to the parametrized device.
    """
    inputs, targets = input_data

    # Instantiate loss
    loss_fn = SLACELoss(**base_config)

    # Move loss_fn (including any buffers/weights) to the specified device
    loss_fn.to(torch_device)

    # Move data to the specified device
    # Detach and re-enable grad to move existing tensor to new device correctly
    inputs_dev = inputs.to(torch_device).detach().requires_grad_(True)
    targets_dev = targets.to(torch_device)

    # If class weights are present, move them too (if not using register_buffer)
    if loss_fn.weight is not None:
        loss_fn.weight = loss_fn.weight.to(torch_device)

    return inputs_dev, targets_dev, loss_fn


# -------------
# --- TESTS ---
# -------------


def test_initialization(base_config, torch_device):
    """Tests that the class is instantiated correctly."""
    loss_fn = SLACELoss(**base_config).to(torch_device)
    assert isinstance(loss_fn, nn.Module)
    assert loss_fn.alpha == base_config["alpha"]


def test_forward_pass_execution(loss_data_on_device):
    """Tests that the forward pass returns a scalar without errors on the specified device."""
    inputs, targets, loss_fn = loss_data_on_device
    loss = loss_fn(inputs, targets)

    assert torch.is_tensor(loss)
    assert loss.dim() == 0  # Should be a scalar (mean)
    assert not torch.isnan(loss)  # Should not be NaN
    assert loss.device == inputs.device


def test_gradient_propagation(loss_data_on_device):
    """Verify that gradients flow backwards on the specified device."""
    inputs, targets, loss_fn = loss_data_on_device

    loss = loss_fn(inputs, targets)
    loss.backward()

    assert inputs.grad is not None
    assert torch.sum(torch.abs(inputs.grad)) > 0
    assert inputs.grad.device == inputs.device


def test_use_logits_flag(base_config, input_data, torch_device):
    """Verifies the difference between receiving logits and receiving probabilities."""
    inputs, targets = input_data
    inputs = inputs.to(torch_device)
    targets = targets.to(torch_device)

    # Case A: use_logits = True (Input is pure logits)
    base_config["use_logits"] = True
    loss_fn_logits = SLACELoss(**base_config).to(torch_device)
    loss_a = loss_fn_logits(inputs, targets)

    # Case B: use_logits = False (Input is probabilities)
    base_config["use_logits"] = False
    loss_fn_probs = SLACELoss(**base_config).to(torch_device)
    probs = F.softmax(inputs, dim=1)  # Apply manual softmax outside
    loss_b = loss_fn_probs(probs, targets)

    assert torch.allclose(loss_a, loss_b, atol=1e-6)


def test_class_weights(base_config, input_data, torch_device):
    """Verifies that passing class weights alters the result."""
    inputs, targets = input_data
    inputs = inputs.to(torch_device)
    targets = targets.to(torch_device)

    # Without weights
    loss_fn_no_weight = SLACELoss(**base_config).to(torch_device)
    loss_nw = loss_fn_no_weight(inputs, targets)

    # With weights (0 for all classes to nullify loss)
    weights = torch.zeros(base_config["num_classes"])
    loss_fn_weighted = SLACELoss(**base_config, weight=weights).to(torch_device)
    loss_w = loss_fn_weighted(inputs, targets)

    # With weights (1 for all classes, should be same as no weights)
    weights_ones = torch.ones(base_config["num_classes"])
    loss_fn_weighted_ones = SLACELoss(**base_config, weight=weights_ones).to(
        torch_device
    )
    loss_w_ones = loss_fn_weighted_ones(inputs, targets)

    assert loss_w.item() == 0.0
    assert loss_nw.item() > 0.0
    assert torch.allclose(loss_nw, loss_w_ones, atol=1e-6)


@pytest.mark.parametrize("targets_type", [torch.float, torch.long])
def test_target_dtype_mismatch(base_config, input_data, torch_device, targets_type):
    """
    Tests passing different target dtypes (Float and Long) on the specified device.
    The loss should be robust enough to handle the float target by casting it to
    long internally.
    """
    inputs, targets = input_data

    inputs = inputs.to(torch_device).detach().requires_grad_(True)
    targets = targets.to(torch_device)

    if targets_type == torch.float:
        targets = targets.float()

    loss_fn = SLACELoss(**base_config).to(torch_device)

    try:
        loss = loss_fn(inputs, targets)
        assert not torch.isnan(loss)
    except RuntimeError as e:
        if targets_type == torch.float:
            pytest.fail(
                f"Loss failed with targets type {targets_type}. Target should be cast to Long internally: {e}"
            )
        raise e


def test_input_target_device_mismatch(base_config, input_data):
    """Verify that passing targets on CPU while inputs are on GPU raises RuntimeError."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    loss_fn = SLACELoss(**base_config)
    inputs, targets = input_data

    inputs_gpu = inputs.to("cuda")

    with pytest.raises(RuntimeError, match="be on the same device"):
        loss_fn(inputs_gpu, targets)


def test_target_shape_mismatch(base_config, torch_device):
    """
    Tests targets as [B, 1] instead of [B] on the specified device.
    """
    loss_fn = SLACELoss(**base_config).to(torch_device)
    inputs = torch.randn(4, 5, device=torch_device)

    # Target with an extra dimension [Batch, 1]
    targets_unsqueezed = torch.randint(0, 5, (4, 1), device=torch_device)

    try:
        loss_fn(inputs, targets_unsqueezed)
    except Exception as e:
        pytest.fail(
            f"Code does not handle targets with shape [Batch, 1] on {torch_device}: {e}"
        )


def test_numerical_stability_high_alpha(base_config, input_data, torch_device):
    """
    Very high Alpha can cause instability in exp/softmax on the specified device.
    """
    base_config["alpha"] = 1000.0  # Extreme value
    loss_fn = SLACELoss(**base_config).to(torch_device)

    inputs, targets = input_data
    inputs = inputs.to(torch_device)
    targets = targets.to(torch_device)

    loss = loss_fn(inputs, targets)
    assert not torch.isnan(loss), f"High alpha caused NaN on {torch_device}"


def test_relative_sensitivity_improvement(base_config, input_data, torch_device):
    """
    Verifies that increasing the probability in the target class reduces the loss.
    Loss(B) < Loss(A).
    """
    inputs, targets = input_data
    device = torch_device

    # Isolate one example for clear analysis
    input_A = inputs[0:1].to(device)
    target_A = targets[0:1].to(device)
    target_idx = target_A.item()

    loss_fn = SLACELoss(**base_config).to(device)

    # Condition A: Base loss (random inputs)
    loss_A = loss_fn(input_A, target_A)

    # Condition B: Optimized inputs. Artificially boost the logit
    # of the target class to force an "improvement."
    input_B = input_A.clone()
    input_B[0, target_idx] += 10.0  # Large confidence boost

    loss_B = loss_fn(input_B, target_A)

    # The loss with better confidence (B) must be smaller than the base loss (A)
    assert loss_B.item() < loss_A.item(), (
        f"Loss did not decrease when prediction improved. Loss A: {loss_A.item()},"
        f" Loss B: {loss_B.item()}"
    )


def test_relative_sensitivity_deterioration(base_config, input_data, torch_device):
    """
    Verifies that increasing the probability in an incorrect class far from the target increases the loss.
    Loss(B) > Loss(A).
    """
    inputs, targets = input_data
    device = torch_device

    input_A = inputs[0:1].to(device)
    target_A = targets[0:1].to(device)
    target_idx = target_A.item()
    num_classes = base_config["num_classes"]

    # Find the index farthest from the target (corner cases for ordinal loss)
    if target_idx < num_classes / 2:
        incorrect_idx = num_classes - 1
    else:
        incorrect_idx = 0

    # Ensure the incorrect index is not the target index
    if incorrect_idx == target_idx:
        incorrect_idx = (incorrect_idx + 1) % num_classes

    loss_fn = SLACELoss(**base_config).to(device)

    # Condition A: Base loss
    loss_A = loss_fn(input_A, target_A)

    # Condition B: Worse inputs. Artificially boost the logit
    # of the incorrect, far-away class.
    input_B = input_A.clone()
    input_B[0, incorrect_idx] += 10.0  # Large confidence boost towards error

    loss_B = loss_fn(input_B, target_A)

    # The loss with worse confidence (B) must be greater than the base loss (A)
    assert loss_B.item() > loss_A.item(), (
        f"Loss did not increase when prediction worsened. Loss A: {loss_A.item()},"
        f" Loss B: {loss_B.item()}"
    )


def test_relative_sensitivity_invariance(base_config, input_data, torch_device):
    """
    Verifies that adding a constant (C) to all logits does not change the loss
    (due to Softmax invariance to translation). This only applies if use_logits=True.
    """
    inputs, targets = input_data
    device = torch_device

    input_A = inputs[0:1].to(device)
    target_A = targets[0:1].to(device)

    loss_fn = SLACELoss(**base_config).to(device)

    # Condition A: Base loss
    loss_A = loss_fn(input_A, target_A)

    # Condition B: Translated inputs. Add a constant to all logits.
    # Softmax(A + C) = Softmax(A)
    input_B = input_A.clone()
    input_B += 5.0  # Add a high constant to all logits

    loss_B = loss_fn(input_B, target_A)

    # The losses must be almost identical
    assert torch.allclose(loss_A, loss_B, atol=1e-6), (
        f"Loss changed due to logit translation. Loss A: {loss_A.item()},"
        f" Loss B: {loss_B.item()}"
    )


def test_exact_value_computation(base_config, torch_device):
    """
    Verifies that the loss computation matches a manual calculation for a simple case.
    """

    # Simple case: 3 classes, target class 1
    base_config["num_classes"] = 3
    loss_fn = SLACELoss(**base_config).to(torch_device)
    input_simple = torch.tensor([[2.0, 1.0, 0.1]], device=torch_device)
    target_simple = torch.tensor([1], device=torch_device)
    loss = loss_fn(input_simple, target_simple).item()

    expected = 0.8163748706490453

    assert torch.isclose(
        torch.tensor(loss), torch.tensor(expected), atol=1e-6
    ), f"Loss computation mismatch. Expected: {expected}, Got: {loss.item()}"
