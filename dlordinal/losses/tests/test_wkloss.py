import numpy as np
import pytest
import torch

from ..losses import WKLoss  # Asegúrate de importar correctamente tu módulo


def test_wkloss_creation():
    loss = WKLoss(num_classes=6)
    assert isinstance(loss, WKLoss)


def test_wkloss_basic():
    num_classes = 6
    penalization_type = "quadratic"

    loss = WKLoss(num_classes, penalization_type)

    input_data = torch.tensor(
        [
            [-2.4079, -2.5133, -2.6187, -2.0652, -3.7299, -5.1068],
            [-2.4079, -2.1725, -2.1459, -3.3318, -3.9624, -4.4700],
            [-2.4079, -1.7924, -2.0101, -4.1030, -3.3445, -4.4812],
        ]
    )

    target = torch.tensor([2, 2, 1])

    # Compute the loss
    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() > 0


def test_wkloss_custom_weight():
    num_classes = 6
    penalization_type = "quadratic"
    weight = np.array(
        [1.60843373, 0.55394191, 1.02692308, 0.78070175, 1.12184874, 2.34210526]
    )

    loss = WKLoss(num_classes, penalization_type, weight)

    input_data = torch.tensor(
        [
            [-2.4079, -2.5133, -1.6020, -2.2353, -3.2371, -4.6382],
            [-2.4079, -2.5133, -2.6187, -2.6082, -1.8703, -2.4251],
            [-2.4079, -1.8561, -3.9906, -5.2992, -5.8944, -7.1456],
        ]
    )

    target = torch.tensor([2, 4, 1])

    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() > 0


if __name__ == "__main":
    test_wkloss_creation()
    test_wkloss_basic()
    test_wkloss_custom_weight()
