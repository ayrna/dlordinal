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


def test_clm_thresholds_exhaustive():
    num_classes = 5
    link_function = "logit"
    min_distance = 0.0

    clm = CLM(
        num_classes=num_classes, link_function=link_function, min_distance=min_distance
    )

    lambdas_1 = torch.Tensor([0.2, 0.05, 0.7])
    th_base_1 = torch.Tensor([0.5])
    th_1 = clm._convert_thresholds(th_base_1, lambdas_1, min_distance)
    expected_th_1 = torch.Tensor([0.5, 0.54, 0.5425, 1.0325])
    assert torch.allclose(th_1, expected_th_1)

    lambdas_2 = torch.Tensor([7.2, 0.4, 4.3])
    th_base_2 = torch.Tensor([1.2])
    th_2 = clm._convert_thresholds(th_base_2, lambdas_2, min_distance)
    expected_th_2 = torch.Tensor([1.2, 53.04, 53.2, 71.69])
    assert torch.allclose(th_2, expected_th_2)

    lambdas_3 = torch.Tensor([-5.6, 6.1, -8.9])
    th_base_3 = torch.Tensor([-0.7])
    th_3 = clm._convert_thresholds(th_base_3, lambdas_3, min_distance)
    expected_th_3 = torch.Tensor([-0.7, 30.66, 67.87, 147.08])
    assert torch.allclose(th_3, expected_th_3)


def test_clm_probas_from_projection_and_thresholds():
    num_classes = 5
    link_function = "logit"
    min_distance = 0.0

    clm = CLM(
        num_classes=num_classes, link_function=link_function, min_distance=min_distance
    )

    th = torch.Tensor([0, 1, 2, 3])
    wx = torch.Tensor([2.2])
    probas = clm._clm(wx, th)
    expected_probas = torch.Tensor(
        [
            0.0997504891196851,
            0.131724727381297,
            0.21869078618654,
            0.23980847844009,
            0.310025518872387,
        ]
    )
    assert torch.allclose(probas, expected_probas, atol=1e-06)

    th = torch.Tensor([0.5, 0.54, 0.5425, 1.0325])
    wx = torch.Tensor([0.25])
    probas = clm._clm(wx, th)
    expected_probas = torch.Tensor(
        [
            0.562176500885798,
            0.00981963204572056,
            0.0006119309317415,
            0.113610607034421,
            0.313781329102319,
        ]
    )
    assert torch.allclose(probas, expected_probas, atol=1e-06)


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
