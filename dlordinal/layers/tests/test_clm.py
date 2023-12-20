import torch

from ..clm import CLM


def test_clm_creation():
    input_shape = 6
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
