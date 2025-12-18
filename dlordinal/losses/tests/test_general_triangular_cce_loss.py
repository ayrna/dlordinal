import numpy as np
import pytest
import torch

from dlordinal.losses import GeneralTriangularCrossEntropyLoss

# TODO: remove in 3.0.0
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_general_triangular_loss_creation():
    # Generate 12 random values between 0.01 and 0.2
    alphas = np.array(
        [
            0.11073934,
            0.10117434,
            0.01966654,
            0.04475356,
            0.02453978,
            0.07121018,
            0.1535088,
            0.01083356,
            0.05561108,
            0.04605048,
            0.02095387,
            0.15887076,
        ]
    )

    num_classes = 6

    loss = GeneralTriangularCrossEntropyLoss(num_classes=num_classes, alphas=alphas)
    assert isinstance(loss, GeneralTriangularCrossEntropyLoss)


def test_general_triangular_loss_basic():
    alphas = np.array(
        [
            0.11073934,
            0.10117434,
            0.01966654,
            0.04475356,
            0.02453978,
            0.07121018,
            0.1535088,
            0.01083356,
            0.05561108,
            0.04605048,
            0.02095387,
            0.15887076,
        ]
    )

    num_classes = 6

    loss = GeneralTriangularCrossEntropyLoss(num_classes=num_classes, alphas=alphas)

    input_data = torch.tensor(
        [
            [-1.6488, -2.5838, -2.8312, -1.9495, -2.4759, -3.4682],
            [-1.7872, -3.9560, -6.2586, -8.3967, -7.9779, -8.0079],
            [-2.4078, -2.5133, -2.5584, -1.7485, -2.3675, -2.6099],
        ]
    )
    target = torch.tensor([4, 0, 5])

    # Compute the loss
    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() > 0


def test_general_triangular_loss_exactvalue():
    alphas = np.array(
        [
            0.11073934,
            0.10117434,
            0.01966654,
            0.04475356,
            0.02453978,
            0.07121018,
            0.1535088,
            0.01083356,
            0.05561108,
            0.04605048,
            0.02095387,
            0.15887076,
        ]
    )

    num_classes = 6

    loss = GeneralTriangularCrossEntropyLoss(num_classes=num_classes, alphas=alphas)

    input_data = torch.tensor(
        [
            [-1.6488, -2.5838, -2.8312, -1.9495, -2.4759, -3.4682],
            [-1.7872, -3.9560, -6.2586, -8.3967, -7.9779, -8.0079],
            [-2.4078, -2.5133, -2.5584, -1.7485, -2.3675, -2.6099],
        ]
    )
    target = torch.tensor([4, 0, 5])

    # Compute the loss
    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() == pytest.approx(1.3814, rel=1e-3)


def test_general_triangular_loss_relative():
    alphas = np.array(
        [
            0.11073934,
            0.10117434,
            0.01966654,
            0.04475356,
            0.02453978,
            0.07121018,
            0.1535088,
            0.01083356,
            0.05561108,
            0.04605048,
            0.02095387,
            0.15887076,
        ]
    )

    num_classes = 6

    loss = GeneralTriangularCrossEntropyLoss(num_classes=num_classes, alphas=alphas)

    input_data = torch.tensor(
        [
            [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    input_data2 = torch.tensor(
        [
            [0.0, 0.0, 0.0, 100.0, 0.0, 0.0],
        ]
    )
    input_data3 = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 100.0, 0.0],
        ]
    )
    target = torch.tensor([1])

    # Compute the loss
    output = loss(input_data, target)
    output2 = loss(input_data2, target)
    output3 = loss(input_data3, target)

    assert output3.item() >= output2.item() > output.item()


def test_general_triangular_loss_same_alphas():
    alphas = np.array(
        [
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
        ]
    )

    num_classes = 6

    loss = GeneralTriangularCrossEntropyLoss(num_classes=num_classes, alphas=alphas)

    input_data = torch.tensor(
        [
            [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    input_data2 = torch.tensor(
        [
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
        ]
    )
    target = torch.tensor([1])

    # Compute the loss
    output = loss(input_data, target)
    output2 = loss(input_data2, target)

    assert output2.item() == output.item()


def test_general_triangular_loss_different_alphas():
    alphas = np.array(
        [
            0.05,
            0.05,
            0.05,
            0.05,
            0.15,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
        ]
    )

    num_classes = 6

    loss = GeneralTriangularCrossEntropyLoss(num_classes=num_classes, alphas=alphas)

    input_data = torch.tensor(
        [
            [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    input_data2 = torch.tensor(
        [
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
        ]
    )
    target = torch.tensor([1])

    # Compute the loss
    output = loss(input_data, target)
    output2 = loss(input_data2, target)

    assert output2.item() < output.item()
