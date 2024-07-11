import warnings

import numpy as np
import pytest
import torch

from dlordinal.output_layers import CLM


def _test_probas(clm):
    projections = torch.rand(32, 1)
    probas = clm(projections)
    total_probas = torch.sum(probas, dim=1)
    assert torch.allclose(total_probas, torch.ones_like(total_probas))
    assert isinstance(probas, torch.Tensor)

    return projections, probas, total_probas


def test_clm_creation():
    num_classes = 3
    link_function = "logit"
    min_distance = 0.0

    clm = CLM(
        num_classes=num_classes, link_function=link_function, min_distance=min_distance
    )

    assert isinstance(clm, CLM)


def test_clm_probas():
    num_classes = 5
    link_function = "logit"
    min_distance = 0.0

    clm = CLM(
        num_classes=num_classes, link_function=link_function, min_distance=min_distance
    )

    _test_probas(clm)


def test_clm_thresholds():
    num_classes = 5
    link_function = "logit"
    min_distance = 0.0

    clm = CLM(
        num_classes=num_classes, link_function=link_function, min_distance=min_distance
    )

    thresholds = clm._convert_thresholds(
        clm.thresholds_b_, clm.thresholds_a_, min_distance
    )
    expected_thresholds = torch.tensor([float(i) for i in range(num_classes - 2 + 1)])

    assert (
        thresholds.shape[0] == clm.thresholds_b_.shape[0] + clm.thresholds_a_.shape[0]
    )

    assert torch.allclose(thresholds, expected_thresholds)

    _test_probas(clm)


def test_clm_link_functions():
    for link in ["logit", "probit", "cloglog"]:
        for num_classes in range(3, 12):
            clm = CLM(num_classes=num_classes, link_function=link, min_distance=0.0)
            assert clm.link_function == link
            assert clm.num_classes == num_classes

            _test_probas(clm)


def test_clm_all_combinations():
    for link in ["logit", "probit", "cloglog"]:
        for num_classes in range(3, 12):
            for min_distance in np.linspace(0.0, 0.1, 10):
                clm = CLM(
                    num_classes=num_classes,
                    link_function=link,
                    min_distance=min_distance,
                )
                assert clm.link_function == link
                assert clm.num_classes == num_classes
                assert clm.min_distance == min_distance

                _test_probas(clm)


def test_clm_clip():
    input_shape = 12
    num_classes = 6
    link_function = "cloglog"
    min_distance = 0.0

    clm = CLM(
        num_classes=num_classes,
        link_function=link_function,
        min_distance=min_distance,
        clip_warning=True,
    )
    input_data = torch.rand(8, input_shape) * 100
    with pytest.warns(Warning, match="Clipping"):
        clm(input_data)

    warnings.filterwarnings("error")
    clm(input_data)
    _test_probas(clm)

    clm = CLM(
        num_classes=num_classes,
        link_function=link_function,
        min_distance=min_distance,
        clip_warning=False,
    )
    clm(input_data)
    _test_probas(clm)

    clm = CLM(
        num_classes=num_classes,
        link_function=link_function,
        min_distance=min_distance,
        clip_warning=True,
    )
    input_data = torch.rand(8, input_shape) * 0.1
    clm(input_data)
    _test_probas(clm)
    warnings.resetwarnings()
