import pytest
import torch
from torch.nn import functional as F

from dlordinal.losses import BetaLoss, NormalizedCrossEntropyLoss


@pytest.mark.parametrize("smoothing", [0.0, 0.3])
def test_soft_targets_match_weighted_formula_and_gradients(device, smoothing):
    logits = torch.tensor(
        [[-1.0, 0.4, 2.0], [0.2, 0.9, -0.5]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    target = logits.new_tensor([[0.1, 0.7, 0.2], [0.6, 0.4, 0.0]])
    weight = logits.new_tensor([0.5, 2.0, 3.0])
    actual = NormalizedCrossEntropyLoss(weight, label_smoothing=smoothing)(
        logits, target
    )
    q = target * (1 - smoothing) + smoothing / 3
    expected = -(q * weight * logits.log_softmax(1)).sum() / (q * weight).sum()
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 2, 2)])
@pytest.mark.parametrize("smoothing", [0.0, 0.3, 1.0])
def test_hard_and_one_hot_targets_are_equivalent(device, shape, smoothing):
    torch.manual_seed(1)
    logits = torch.randn(shape, dtype=torch.float64, device=device, requires_grad=True)
    class_dim = 0 if len(shape) == 1 else 1
    target_shape = shape[:class_dim] + shape[class_dim + 1 :]
    target = torch.randint(3, target_shape, device=device)
    one_hot = F.one_hot(target, 3).movedim(-1, class_dim).to(logits.dtype)
    weight = logits.new_tensor([0.5, 2.0, 3.0])
    loss = NormalizedCrossEntropyLoss(weight, label_smoothing=smoothing)
    actual = loss(logits, target)
    expected = loss(logits, one_hot)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("target_dtype", [torch.long, torch.uint8])
def test_hard_targets_match_pytorch_without_smoothing(device, weighted, target_dtype):
    logits = torch.tensor(
        [[1.0, -0.2, 0.4], [-1.0, 2.0, 0.1]], device=device, requires_grad=True
    )
    target = torch.tensor([0, 2], dtype=target_dtype, device=device)
    weight = logits.new_tensor([0.5, 2.0, 3.0]) if weighted else None
    actual = NormalizedCrossEntropyLoss(weight)(logits, target)
    expected = F.cross_entropy(logits, target, weight=weight)
    torch.testing.assert_close(actual, expected)


def test_unsigned_targets_preserve_negative_ignore_index_semantics(device):
    logits = torch.zeros(1, 200, device=device)
    target = torch.tensor([156], dtype=torch.uint8, device=device)
    actual = NormalizedCrossEntropyLoss()(logits, target)
    torch.testing.assert_close(actual, F.cross_entropy(logits, target))


@pytest.mark.parametrize("scale", [1e-20, 7.0])
def test_weight_scale_preserves_loss_and_gradients(device, scale):
    logits = torch.tensor(
        [[0.2, 0.8, -0.3], [-0.4, 0.1, 1.3]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    target = logits.new_tensor([[0.2, 0.5, 0.3], [0.0, 0.3, 0.7]])
    weight = logits.new_tensor([0.5, 2.0, 3.0])
    actual = NormalizedCrossEntropyLoss(weight * scale, label_smoothing=0.2)(
        logits, target
    )
    expected = NormalizedCrossEntropyLoss(weight, label_smoothing=0.2)(logits, target)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("scale", [1e-40, 1e38])
def test_extreme_common_weight_scale_preserves_values_and_gradients(device, scale):
    logits = torch.tensor(
        [[0.2, 0.8, -0.3], [-0.4, 0.1, 1.3]],
        device=device,
        requires_grad=True,
    )
    target = logits.new_tensor([[0.2, 0.5, 0.3], [0.0, 0.3, 0.7]])
    weight = logits.new_tensor([1.0, 2.0, 3.0])
    actual = NormalizedCrossEntropyLoss(weight * scale, label_smoothing=0.2)(
        logits, target
    )
    expected = NormalizedCrossEntropyLoss(weight, label_smoothing=0.2)(logits, target)
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=1e-6)
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-5, atol=1e-6)


@pytest.mark.parametrize("soft", [False, True])
def test_absent_large_weight_preserves_small_active_weights(device, soft):
    logits = torch.tensor(
        [[0.2, 0.8, -0.3], [-0.4, 0.1, 1.3]],
        device=device,
        requires_grad=True,
    )
    target = torch.tensor([1, 2], device=device)
    q = F.one_hot(target, 3).to(torch.float64)
    if soft:
        target = q.to(logits.dtype)
    weight = logits.new_tensor([1e38, 1e-40, 2e-40])
    actual = NormalizedCrossEntropyLoss(weight)(logits, target)
    weighted_target = q * weight.double()
    expected = -(weighted_target * logits.double().log_softmax(1)).sum()
    expected = expected / weighted_target.sum()
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-7)
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_soft_target_gradients_include_zero_mass_classes(device):
    logits = torch.tensor([[0.2, 0.8, -0.3]], dtype=torch.float64, device=device)
    target = logits.new_tensor([[0.0, 0.3, 0.7]], requires_grad=True)
    weight = logits.new_tensor([1.0, 2.0, 3.0])
    actual = NormalizedCrossEntropyLoss(weight)(logits, target)
    expected = (
        -(target * weight * logits.log_softmax(1)).sum() / (target * weight).sum()
    )
    actual_grad = torch.autograd.grad(actual, target, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, target)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("kind", ["zero_weights", "zero_targets", "ignored"])
def test_zero_effective_weight_has_zero_loss_and_gradients(device, kind):
    logits = torch.randn(2, 3, device=device, requires_grad=True)
    target = torch.tensor([0, 2], device=device)
    weight = None
    if kind == "zero_weights":
        weight = logits.new_zeros(3)
    elif kind == "zero_targets":
        target = torch.zeros_like(logits)
    else:
        target = torch.full_like(target, -100)
    loss = NormalizedCrossEntropyLoss(weight)(logits, target)
    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_ignored_targets_have_zero_gradients_with_smoothing(device):
    logits = torch.tensor(
        [[1.0, -0.2, 0.4], [-1.0, 2.0, 0.1]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    target = torch.tensor([-100, 2], device=device)
    weight = logits.new_tensor([0.5, 2.0, 3.0])
    loss = NormalizedCrossEntropyLoss(weight, label_smoothing=0.3)
    actual = loss(logits, target)
    expected = loss(logits[1:], target[1:])
    torch.testing.assert_close(actual, expected)
    actual.backward()
    torch.testing.assert_close(logits.grad[0], torch.zeros_like(logits.grad[0]))


@pytest.mark.parametrize("reduction", ["none", "sum"])
@pytest.mark.parametrize("soft", [False, True])
def test_other_reductions_follow_pytorch(device, reduction, soft):
    logits = torch.tensor([[1.0, 0.2, -0.4], [-1.0, 0.0, 0.5]], device=device)
    target = torch.tensor([0, 2], device=device)
    if soft:
        target = F.one_hot(target, 3).to(logits.dtype)
    weight = logits.new_tensor([0.5, 2.0, 3.0])
    loss = NormalizedCrossEntropyLoss(weight, reduction=reduction, label_smoothing=0.2)
    expected = F.cross_entropy(
        logits, target, weight=weight, reduction=reduction, label_smoothing=0.2
    )
    torch.testing.assert_close(loss(logits, target), expected)


def test_class_weights_follow_module_precision_and_device(device):
    loss = NormalizedCrossEntropyLoss(torch.tensor([0.5, 2.0, 3.0])).to(
        device=device, dtype=torch.float64
    )
    assert loss.weight.dtype == torch.float64
    assert loss.weight.device == device
    assert "weight" in dict(loss.named_buffers())
    logits = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64, device=device)
    target = torch.tensor([1], device=device)
    assert loss(logits, target).dtype == torch.float64


def test_beta_loss_composes_with_normalized_base_loss(device):
    logits = torch.tensor([[0.1, 0.2, 0.3], [1.0, -0.2, 0.4]], device=device)
    target = torch.tensor([0, 2], device=device)
    weight = logits.new_tensor([0.5, 2.0, 3.0])
    loss = BetaLoss(NormalizedCrossEntropyLoss(weight), num_classes=3, eta=0.4).to(
        device
    )
    q = 0.6 * F.one_hot(target, 3) + 0.4 * loss.cls_probs[target]
    expected = -(q * weight * logits.log_softmax(1)).sum() / (q * weight).sum()
    torch.testing.assert_close(loss(logits, target), expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("soft", [False, True])
def test_low_precision_large_batch_mean_stays_finite(device, dtype, weighted, soft):
    logits = torch.zeros(40000, 10, dtype=dtype, device=device, requires_grad=True)
    target = torch.zeros(40000, dtype=torch.long, device=device)
    if soft:
        target = F.one_hot(target, 10).to(dtype)
    weight = torch.ones(10, dtype=dtype, device=device) if weighted else None
    actual = NormalizedCrossEntropyLoss(weight)(logits, target)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, actual.new_tensor(10.0).log())
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_low_precision_tiny_weights_use_float32_accumulation(device):
    logits = torch.tensor(
        [[0.2, 0.8, -0.3], [-0.4, 0.1, 1.3]],
        dtype=torch.float16,
        device=device,
        requires_grad=True,
    )
    target = torch.tensor([0, 2], device=device)
    weight = logits.new_tensor([1e-6, 2e-6, 3e-6])
    actual = NormalizedCrossEntropyLoss(weight, label_smoothing=0.3)(logits, target)
    expected = NormalizedCrossEntropyLoss(weight.float(), label_smoothing=0.3)(
        logits.float(), target
    )
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize(
    "weight",
    [
        torch.ones(2, 2),
        torch.empty(0),
        torch.tensor([1.0, -1.0]),
        torch.tensor([1.0, float("inf")]),
        torch.tensor([float("nan"), 1.0]),
    ],
)
def test_invalid_weights_raise(weight):
    with pytest.raises(ValueError, match="weight"):
        NormalizedCrossEntropyLoss(weight)
