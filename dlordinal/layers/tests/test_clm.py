import warnings

import pytest
import torch
from dlordinal.layers import CLM


def test_clm_creation():
    num_classes = 3
    link_function = "logit"
    min_distance = 0.0

    clm = CLM(
        num_classes=num_classes, link_function=link_function, min_distance=min_distance
    )

    assert isinstance(clm, CLM)


def test_clm_logit():
    input_shape = 10
    num_classes = 5
    link_function = "logit"
    min_distance = 0.0

    clm = CLM(
        num_classes=num_classes, link_function=link_function, min_distance=min_distance
    )
    input_data = torch.rand(32, input_shape)
    output = clm(input_data)

    assert isinstance(output, torch.Tensor)
    assert clm.link_function == "logit"


def test_clm_probit():
    input_shape = 8
    num_classes = 4
    link_function = "probit"
    min_distance = 0.0

    clm = CLM(
        num_classes=num_classes, link_function=link_function, min_distance=min_distance
    )
    input_data = torch.rand(16, input_shape)
    output = clm(input_data)

    assert isinstance(output, torch.Tensor)
    assert clm.link_function == "probit"


def test_clm_cloglog():
    input_shape = 12
    num_classes = 6
    link_function = "cloglog"
    min_distance = 0.0

    clm = CLM(
        num_classes=num_classes, link_function=link_function, min_distance=min_distance
    )
    input_data = torch.rand(8, input_shape)
    output = clm(input_data)

    assert isinstance(output, torch.Tensor)
    assert clm.link_function == "cloglog"


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

    clm = CLM(
        num_classes=num_classes,
        link_function=link_function,
        min_distance=min_distance,
        clip_warning=False,
    )
    clm(input_data)

    clm = CLM(
        num_classes=num_classes,
        link_function=link_function,
        min_distance=min_distance,
        clip_warning=True,
    )
    input_data = torch.rand(8, input_shape) * 0.1
    clm(input_data)
    warnings.resetwarnings()
