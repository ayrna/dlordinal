import pytest
import torch

from dlordinal.losses import CDWCELoss


@pytest.fixture
def ordinal_batch(device):
    logits = torch.tensor(
        [
            [0.4, -0.5, 0.9, 0.1],
            [-0.3, 0.8, 0.2, -0.4],
            [0.7, 0.3, -0.2, 0.5],
            [0.1, -0.7, 0.4, 0.6],
        ],
        dtype=torch.float64,
        device=device,
    )
    targets = torch.tensor([0, 1, 1, 3], device=device)
    return logits, targets


def reference_sample_losses(logits, targets, alpha, margin, normalize_distance):
    """Evaluate the probability-space definition independently for each sample."""
    probabilities = logits.softmax(dim=1)
    samples = []
    for probability, target in zip(probabilities, targets):
        distances = [abs(j - target.item()) ** alpha for j in range(logits.shape[1])]
        terms = [
            -distance * (1 - probability[j] + margin).clamp(max=1).log()
            for j, distance in enumerate(distances)
        ]
        sample_loss = torch.stack(terms).sum()
        if normalize_distance and sum(distances):
            sample_loss = sample_loss / sum(distances)
        samples.append(sample_loss)
    return torch.stack(samples)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("margin", [0.0, 0.2])
@pytest.mark.parametrize("normalize_distance", [False, True])
def test_weighted_mean_matches_probability_formula(
    ordinal_batch, alpha, margin, normalize_distance
):
    logits, targets = ordinal_batch
    weights = logits.new_tensor([2.0, 9.0, 4.0, 1.0])
    sample_losses = reference_sample_losses(
        logits, targets, alpha, margin, normalize_distance
    )
    selected_weights = weights[targets]
    expected = (selected_weights * sample_losses).sum() / selected_weights.sum()

    actual = CDWCELoss(
        4,
        alpha,
        weights,
        margin,
        reduction="weighted_mean",
        normalize_distance=normalize_distance,
    )(logits, targets)

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("normalize_distance", [False, True])
@pytest.mark.parametrize("with_weights", [False, True])
def test_sample_reductions_retain_global_class_weight_normalization(
    ordinal_batch, normalize_distance, with_weights
):
    logits, targets = ordinal_batch
    weights = logits.new_tensor([2.0, 9.0, 4.0, 1.0]) if with_weights else None
    expected = reference_sample_losses(logits, targets, 1.0, 0.2, normalize_distance)
    if weights is not None:
        expected = expected * weights[targets] / weights.sum()

    results = {}
    for reduction in ("none", "sum", "mean"):
        results[reduction] = CDWCELoss(
            4,
            1.0,
            weights,
            0.2,
            reduction=reduction,
            normalize_distance=normalize_distance,
        )(logits, targets)

    assert results["none"].shape == targets.shape
    torch.testing.assert_close(results["none"], expected, rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(results["sum"], expected.sum(), rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(results["mean"], expected.mean(), rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("normalize_distance", [False, True])
def test_weighted_mean_without_weights_equals_mean(ordinal_batch, normalize_distance):
    logits, targets = ordinal_batch
    expected = CDWCELoss(4, normalize_distance=normalize_distance)(logits, targets)
    actual = CDWCELoss(
        4, reduction="weighted_mean", normalize_distance=normalize_distance
    )(logits, targets)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("normalize_distance", [False, True])
@pytest.mark.parametrize("constant", [1.0, 7.0, 1e-12])
def test_constant_weights_recover_unweighted_values_and_gradients(
    ordinal_batch, normalize_distance, constant
):
    logits, targets = ordinal_batch
    logits.requires_grad_()
    expected = CDWCELoss(4, normalize_distance=normalize_distance)(logits, targets)
    actual = CDWCELoss(
        4,
        weight=logits.new_full((4,), constant),
        reduction="weighted_mean",
        normalize_distance=normalize_distance,
    )(logits, targets)
    expected_gradient = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual, logits)[0]

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("normalize_distance", [False, True])
@pytest.mark.parametrize("scale", [1e-40, 17.0, 1e38])
def test_common_weight_scaling_preserves_loss_and_input_gradients(
    ordinal_batch, normalize_distance, scale
):
    logits, targets = ordinal_batch
    logits = logits.float().requires_grad_()
    weights = logits.new_tensor([1.0, 2.0, 3.0, 1.0])
    expected = CDWCELoss(
        4,
        weight=weights,
        reduction="weighted_mean",
        normalize_distance=normalize_distance,
    )(logits, targets)
    actual = CDWCELoss(
        4,
        weight=weights * scale,
        reduction="weighted_mean",
        normalize_distance=normalize_distance,
    )(logits, targets)
    expected_gradient = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual, logits)[0]

    assert torch.isfinite(actual_gradient).all()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-5, atol=1e-7)


def test_distance_normalization_is_per_sample(ordinal_batch):
    logits, targets = ordinal_batch
    weights = logits.new_tensor([2.0, 9.0, 4.0, 1.0])
    selected_weights = weights[targets]
    sample_losses = reference_sample_losses(logits, targets, 1.0, 0.0, False)
    distance_sums = logits.new_tensor([6.0, 4.0, 4.0, 6.0])
    expected = (selected_weights * sample_losses / distance_sums).sum()
    expected = expected / selected_weights.sum()
    global_distance_average = (selected_weights * sample_losses).sum()
    global_distance_average = (
        global_distance_average / (selected_weights * distance_sums).sum()
    )

    actual = CDWCELoss(
        4, 1.0, weights, reduction="weighted_mean", normalize_distance=True
    )(logits, targets)

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)
    assert not torch.isclose(actual, global_distance_average, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("normalize_distance", [False, True])
@pytest.mark.parametrize("weights", [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]])
def test_zero_target_weight_batch_returns_zero_with_zero_gradients(
    ordinal_batch, normalize_distance, weights
):
    logits, targets = ordinal_batch
    logits = (logits * 10000).requires_grad_()
    actual = CDWCELoss(
        4,
        weight=logits.new_tensor(weights),
        reduction="weighted_mean",
        normalize_distance=normalize_distance,
    )(logits, targets)
    actual.backward()

    assert actual.requires_grad
    assert actual.item() == 0.0
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), rtol=0, atol=0)


def test_zero_weight_samples_with_extreme_logits_are_ignored(ordinal_batch):
    logits, targets = ordinal_batch
    logits[0] *= 10000
    logits.requires_grad_()
    weights = logits.new_tensor([0.0, 2.0, 0.0, 1.0])
    loss_fn = CDWCELoss(
        4, weight=weights, reduction="weighted_mean", normalize_distance=True
    )
    actual = loss_fn(logits, targets)
    expected = loss_fn(logits[1:], targets[1:])
    actual.backward()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(
        logits.grad[0], torch.zeros_like(logits[0]), rtol=0, atol=0
    )


@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.int64])
def test_weighted_mean_avoids_low_precision_and_integer_sum_overflow(
    ordinal_batch, weight_dtype
):
    logits, targets = ordinal_batch
    logits = logits.float().requires_grad_()
    values = [60000, 30000, 60000, 60000]
    if weight_dtype == torch.int64:
        values = [6 * 10**18, 3 * 10**18, 6 * 10**18, 6 * 10**18]
    weights = torch.tensor(values, dtype=weight_dtype, device=logits.device)
    actual = CDWCELoss(4, weight=weights, reduction="weighted_mean")(logits, targets)
    expected = CDWCELoss(
        4, weight=logits.new_tensor([2.0, 1.0, 2.0, 2.0]), reduction="weighted_mean"
    )(logits, targets)
    actual_gradient = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, logits)[0]

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_unobserved_large_weight_does_not_change_batch_normalization(ordinal_batch):
    logits, targets = ordinal_batch
    logits = logits.float().requires_grad_()
    actual = CDWCELoss(
        4,
        weight=logits.new_tensor([1e-40, 2e-40, 1e38, 1e-40]),
        reduction="weighted_mean",
    )(logits, targets)
    expected = CDWCELoss(
        4, weight=logits.new_tensor([1.0, 2.0, 3.0, 1.0]), reduction="weighted_mean"
    )(logits, targets)
    actual_gradient = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, logits)[0]

    assert torch.isfinite(actual_gradient).all()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-5, atol=1e-7)


def test_two_dimensional_targets_keep_argmax_semantics(ordinal_batch):
    logits, targets = ordinal_batch
    soft_targets = torch.full_like(logits, 0.1)
    soft_targets.scatter_(1, targets[:, None], 0.7)
    loss_fn = CDWCELoss(
        4,
        weight=logits.new_tensor([2.0, 9.0, 4.0, 1.0]),
        reduction="weighted_mean",
        normalize_distance=True,
    )
    torch.testing.assert_close(
        loss_fn(logits, soft_targets), loss_fn(logits, targets), rtol=0, atol=0
    )


@pytest.mark.parametrize("with_weights", [False, True])
def test_empty_weighted_mean_returns_connected_zero(device, with_weights):
    logits = torch.empty((0, 4), device=device, requires_grad=True)
    targets = torch.empty((0,), device=device, dtype=torch.long)
    weights = logits.new_ones(4) if with_weights else None
    actual = CDWCELoss(4, weight=weights, reduction="weighted_mean")(logits, targets)
    actual.backward()
    assert actual.item() == 0.0
    assert logits.grad.shape == logits.shape


@pytest.mark.parametrize("reduction", ["invalid", "Mean", None])
def test_invalid_reduction_raises(reduction):
    with pytest.raises(ValueError, match="reduction"):
        CDWCELoss(4, reduction=reduction)


@pytest.mark.parametrize(
    "weight",
    [
        torch.ones(3),
        torch.ones(4, 1),
        torch.tensor(1.0),
        torch.tensor([1.0, -1.0, 1.0, 1.0]),
        torch.tensor([1.0, float("nan"), 1.0, 1.0]),
        torch.tensor([1.0, float("inf"), 1.0, 1.0]),
        torch.ones(4, dtype=torch.complex64),
        [1.0, 1.0, 1.0, 1.0],
    ],
)
def test_weighted_mean_validates_class_weights(weight):
    with pytest.raises(ValueError, match="weight"):
        CDWCELoss(4, weight=weight, reduction="weighted_mean")


def test_new_options_are_keyword_only():
    with pytest.raises(TypeError):
        CDWCELoss(4, 0.5, None, 0.0, "sum")
