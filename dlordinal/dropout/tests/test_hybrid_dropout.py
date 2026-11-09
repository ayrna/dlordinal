import pytest
import torch
from torch import nn

from dlordinal.dropout import HybridDropout, HybridDropoutContainer


class ModelWithHybridDropout(nn.Module):
    def __init__(self):
        super(ModelWithHybridDropout, self).__init__()
        self.fc1 = nn.Linear(10, 256)
        self.hybrid_dropout = HybridDropout()
        self.fc2 = nn.Linear(256, 5)  # Cambiar de 5 a la dimensión correcta

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.hybrid_dropout(x)
        x = self.fc2(x)
        return x


@pytest.fixture
def model():
    return ModelWithHybridDropout()


@pytest.fixture
def hybrid_dropout_container(model):
    return HybridDropoutContainer(model)


def test_set_targets(hybrid_dropout_container, device):
    targets = torch.tensor([1, 2, 3, 4, 5]).to(device)
    hybrid_dropout_container.set_targets(targets)

    hybrid_dropout_container = hybrid_dropout_container.to(device)

    hybrid_dropout = None
    for module in hybrid_dropout_container.model.modules():
        if isinstance(module, HybridDropout):
            hybrid_dropout = module
            hybrid_dropout = hybrid_dropout.to(device)
            break
    assert hybrid_dropout is not None
    assert torch.all(torch.eq(hybrid_dropout.batch_targets, targets))


def test_forward_with_targets(device):
    print(device)
    model = ModelWithHybridDropout().to(device)
    inputs = torch.randn(32, 10).to(device)
    targets = torch.randn(32).to(device)
    model.hybrid_dropout.batch_targets = targets
    outputs = model(inputs)

    # The number of classes is 5
    assert outputs.shape == (32, 5)


def test_wrong_target_shape(hybrid_dropout_container, device):
    inputs = torch.randn(32, 10).to(device)
    targets = torch.randn(32, 5).to(device)  # Wrong shape

    hybrid_dropout_container.set_targets(targets)

    with pytest.raises(ValueError):
        hybrid_dropout_container = hybrid_dropout_container.to(device)
        hybrid_dropout_container(inputs)


def test_forward_without_targets(hybrid_dropout_container, device):
    inputs = torch.randn(32, 10).to(device)

    with pytest.raises(ValueError):
        hybrid_dropout_container = hybrid_dropout_container.to(device)
        hybrid_dropout_container(inputs)


def test_hybrid_dropout_scaling(device):
    torch.manual_seed(42)

    model = ModelWithHybridDropout().to(device)
    model.train()

    # Random input tensor and targets
    x = torch.randn(128, 10, device=device)
    targets = torch.randint(0, 5, (128,), device=device)  # Random targets between 0-4
    model.hybrid_dropout.batch_targets = targets

    # Output before applying dropout
    with torch.no_grad():
        pre_dropout = torch.relu(model.fc1(x))

    # Output after applying hybrid dropout
    with torch.no_grad():
        post_dropout = model.hybrid_dropout(pre_dropout)

    # Check dropout rate corresponds to probability p
    mask = (post_dropout == 0).float()
    drop_rate = mask.mean().item()
    assert 0.2 < drop_rate < 0.8, f"Drop rate fuera de rango: {drop_rate}"

    # Check scaling
    mean_in = pre_dropout.abs().mean().item()
    mean_out = post_dropout.abs().mean().item()

    # If it is correctly scaled, the means should be approximately equal mean_out ≈ mean_in (tolerance 15%)
    assert (
        abs(mean_out / mean_in - 1.0) < 0.15
    ), f"Escalado incorrecto: in={mean_in}, out={mean_out}"


def _loop_correlations(x, targets):
    # Reference implementation: the original per-unit torch.corrcoef loop
    t = targets.reshape(1, -1)
    out = []
    for unit in range(x.shape[1]):
        concat = torch.cat((x[:, unit].reshape(1, -1), t), 0)
        out.append(float(torch.corrcoef(concat)[0, 1]))
    return torch.nan_to_num(torch.tensor(out), nan=0.0)


@pytest.mark.parametrize("targets_dtype", [torch.long, torch.float])
def test_vectorised_correlation_matches_loop(device, targets_dtype):
    torch.manual_seed(0)
    x = torch.randn(64, 300, device=device)
    x[:, 5] = 0.0  # zero-variance unit
    x[:, 7] = 3.0 * x[:, 3] + 1.0  # correlated units
    targets = torch.randint(0, 5, (64,), device=device).to(targets_dtype)
    x[:, 9] = targets.float()  # perfectly correlated unit

    expected = _loop_correlations(x, targets)
    got = HybridDropout._pearson_correlation(x, targets)

    assert got.device == x.device
    assert got.shape == (300,)
    assert torch.allclose(got.cpu(), expected, atol=1e-6)


def test_undefined_correlation_is_neutral(device):
    p, beta = 0.5, 0.4
    neutral = beta * 0.5 + (1 - beta) * p
    dropout = HybridDropout(p=p, beta=beta).to(device)

    # A dead unit (zero variance) has no ordinal preference
    x = torch.randn(32, 4, device=device)
    x[:, 0] = 0.0
    targets = torch.randint(0, 5, (32,), device=device)
    probs = dropout._drop_probabilities(x, targets)
    assert torch.isclose(probs[0].cpu(), torch.tensor(neutral))

    # A batch where all targets are equal has no ordinal information
    targets = torch.full((32,), 2, device=device)
    probs = dropout._drop_probabilities(x, targets)
    assert torch.allclose(probs.cpu(), torch.full((4,), neutral))

    # With beta=1 and constant targets, units must not all be dropped
    dropout = HybridDropout(p=0.5, beta=1.0).to(device).train()
    dropout.batch_targets = torch.full((32,), 2, device=device)
    torch.manual_seed(0)
    kept = sum(
        bool((dropout(x.abs() + 0.1) != 0).any()) for _ in range(20)
    )  # P(all 4 units dropped) = 1/16 per call
    assert kept > 10


def test_per_unit_scaling_preserves_expectation(device):
    torch.manual_seed(0)
    n, c = 256, 8
    base = torch.randn(n, device=device)
    targets = (base > 0).long()
    x = torch.randn(n, c, device=device).abs() + 1.0
    x[:, 0] = 2.0 + base  # positively correlated unit (dropped less often)
    x[:, 1] = 2.0 - base  # negatively correlated unit (dropped more often)

    dropout = HybridDropout(p=0.5, beta=0.5).to(device).train()
    dropout.batch_targets = targets
    probs = dropout._drop_probabilities(x, targets)
    assert probs[0] < 0.4 and probs[1] > 0.6

    repeats = 5000
    acc = torch.zeros(c, device=device)
    with torch.no_grad():
        for _ in range(repeats):
            acc += dropout(x).mean(0)
    ratio = (acc / repeats) / x.mean(0)

    assert torch.allclose(ratio.cpu(), torch.ones(c), atol=0.1), ratio


def test_always_dropped_units_have_finite_gradients(device):
    dropout = HybridDropout(p=1.0, beta=0.0).to(device).train()
    x = torch.randn(16, 6, device=device, requires_grad=True)
    dropout.batch_targets = torch.randint(0, 3, (16,), device=device)
    out = dropout(x)
    out.sum().backward()
    assert torch.all(out == 0)
    assert torch.isfinite(x.grad).all()


def test_mask_is_shared_by_the_batch(device):
    dropout = HybridDropout().to(device).train()
    x = torch.randn(32, 64, device=device).abs() + 0.1
    dropout.batch_targets = torch.randint(0, 5, (32,), device=device)
    zeros = dropout(x) == 0
    assert torch.all(zeros.all(0) | ~zeros.any(0))


@pytest.mark.parametrize("shape", [(8,), (8, 16, 4), (8, 16, 4, 4)])
def test_non_2d_input_raises(device, shape):
    dropout = HybridDropout().to(device).train()
    dropout.batch_targets = torch.randint(0, 5, (8,), device=device)
    with pytest.raises(ValueError, match=r"\(N, features\)"):
        dropout(torch.randn(*shape, device=device))


def test_targets_batch_size_mismatch_raises(device):
    dropout = HybridDropout().to(device).train()
    dropout.batch_targets = torch.randint(0, 5, (7,), device=device)
    with pytest.raises(ValueError):
        dropout(torch.randn(8, 16, device=device))


def test_eval_mode_is_identity(device):
    dropout = HybridDropout().to(device).eval()
    x = torch.randn(8, 16, 4, device=device)
    assert dropout(x) is x
