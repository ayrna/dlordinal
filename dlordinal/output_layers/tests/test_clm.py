import warnings

import numpy as np
import pytest
import torch

from dlordinal.output_layers import CLM


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


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


def test_clm_clip(device):
    input_shape = 12
    num_classes = 6
    link_function = "cloglog"
    min_distance = 0.0

    clm = CLM(
        num_classes=num_classes,
        link_function=link_function,
        min_distance=min_distance,
        clip_warning=True,
    ).to(device)
    input_data = torch.rand(8, input_shape).to(device) * 100
    with pytest.warns(Warning, match="Clipping"):
        clm(input_data)

    warnings.filterwarnings("error")
    clm(input_data)
    _test_probas(clm, device)

    clm = CLM(
        num_classes=num_classes,
        link_function=link_function,
        min_distance=min_distance,
        clip_warning=False,
    ).to(device)
    clm(input_data)
    _test_probas(clm, device)

    clm = CLM(
        num_classes=num_classes,
        link_function=link_function,
        min_distance=min_distance,
        clip_warning=True,
    ).to(device)
    input_data = torch.rand(8, input_shape).to(device) * 0.1
    clm(input_data)
    _test_probas(clm, device)
    warnings.resetwarnings()
