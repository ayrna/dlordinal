import numpy as np
import pytest
import torch

from dlordinal.output_layers import CLM


def _test_probas(clm, device):
    projections = torch.rand(32, 1).to(device)
    probas = clm(projections)
    total_probas = torch.sum(probas, dim=1).to(device)
    assert torch.allclose(total_probas, torch.ones_like(total_probas))
    assert isinstance(probas, torch.Tensor)

    return projections, probas, total_probas


def test_clm_creation(device):
    num_classes = 3
    link_function = "logit"
    min_distance = 0.0

    clm = CLM(
        num_classes=num_classes, link_function=link_function, min_distance=min_distance
    ).to(device)

    assert isinstance(clm, CLM)


def test_clm_probas(device):
    num_classes = 5
    link_function = "logit"
    min_distance = 0.0

    clm = CLM(
        num_classes=num_classes, link_function=link_function, min_distance=min_distance
    ).to(device)

    _test_probas(clm, device)


def test_clm_thresholds(device):
    num_classes = 5
    link_function = "logit"
    min_distance = 0.0

    clm = CLM(
        num_classes=num_classes, link_function=link_function, min_distance=min_distance
    ).to(device)

    thresholds = clm._convert_thresholds(
        clm.thresholds_b_, clm.thresholds_a_, min_distance
    )
    expected_thresholds = torch.tensor(
        [float(i) for i in range(num_classes - 2 + 1)]
    ).to(device)

    assert (
        thresholds.shape[0] == clm.thresholds_b_.shape[0] + clm.thresholds_a_.shape[0]
    )

    assert torch.allclose(thresholds, expected_thresholds)

    _test_probas(clm, device)


def test_clm_link_functions(device):
    for link in ["logit", "probit", "cloglog"]:
        for num_classes in range(3, 12):
            clm = CLM(num_classes=num_classes, link_function=link, min_distance=0.0).to(
                device
            )
            assert clm.link_function == link
            assert clm.num_classes == num_classes

            _test_probas(clm, device)


def test_clm_all_combinations(device):
    for link in ["logit", "probit", "cloglog"]:
        for num_classes in range(3, 12):
            for min_distance in np.linspace(0.0, 0.1, 10):
                clm = CLM(
                    num_classes=num_classes,
                    link_function=link,
                    min_distance=min_distance,
                ).to(device)
                assert clm.link_function == link
                assert clm.num_classes == num_classes
                assert clm.min_distance == min_distance

                _test_probas(clm, device)


def test_clm_cloglog_keeps_small_probabilities():
    clm = CLM(num_classes=4, link_function="cloglog")
    # Initial thresholds are (0, 1, 2)
    probs = clm(torch.tensor([[25.0], [-1.0], [0.0]]))
    assert torch.all(probs > 0)
    assert torch.isfinite(torch.log(probs)).all()
    assert torch.allclose(probs.sum(dim=1), torch.ones(3))
    # Lower tail: P(y = 0) = 1 - exp(-exp(0 - 25)) = -expm1(-exp(-25))
    expected_p0 = -torch.expm1(-torch.exp(torch.tensor(-25.0, dtype=torch.float64)))
    assert probs[0, 0].item() == pytest.approx(expected_p0.item(), rel=1e-5)
    # Upper tail: P(y = 3) = exp(-exp(2 + 1)), about 1.9e-9, below float32 eps
    expected_p3 = torch.exp(-torch.exp(torch.tensor(3.0, dtype=torch.float64)))
    assert probs[1, 3].item() == pytest.approx(expected_p3.item(), rel=1e-5)


def test_clm_probit_matches_normal_cdf():
    torch.manual_seed(0)
    clm = CLM(num_classes=5, link_function="probit")
    x = torch.randn(10, 1)
    thresholds = clm._convert_thresholds(
        clm.thresholds_b_, clm.thresholds_a_, clm.min_distance
    )
    cdf = torch.distributions.Normal(0.0, 1.0).cdf(thresholds[None, :] - x)
    expected = torch.diff(cdf, prepend=torch.zeros(10, 1), append=torch.ones(10, 1))
    assert torch.allclose(clm(x), expected, atol=1e-6)


def test_clm_invalid_link_function():
    with pytest.raises(ValueError, match="link_function"):
        CLM(num_classes=4, link_function="Logit")


def test_stable_sigmoid_matches_torch_sigmoid():
    from dlordinal.output_layers.utils import stable_sigmoid

    t = torch.tensor([-1000.0, -50.0, -1.0, 0.0, 1.0, 50.0, 1000.0])
    assert torch.equal(stable_sigmoid(t), torch.sigmoid(t))
