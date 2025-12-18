import pytest
import torch

from dlordinal.losses import TriangularCrossEntropyLoss

# TODO: remove in 3.0.0
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_triangular_loss_creation():
    loss = TriangularCrossEntropyLoss(num_classes=5)
    assert isinstance(loss, TriangularCrossEntropyLoss)


def test_triangular_loss_basic():
    loss = TriangularCrossEntropyLoss(num_classes=6)

    input_data = torch.tensor(
        [
            [-1.5130, -3.5241, -4.9549, -5.1838, -5.8980, -6.5854],
            [-1.8332, -1.8776, -2.9285, -2.0628, -2.9925, -3.4792],
            [-1.3934, -2.6727, -3.0570, -2.9145, -2.8310, -3.8281],
        ]
    )
    target = torch.tensor([1, 2, 3])

    # Compute the loss
    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() > 0


def test_triangular_loss_exactvalue():
    loss = TriangularCrossEntropyLoss(num_classes=6)

    input_data = torch.tensor(
        [
            [0.8, 0.1, 0.1, 0.0, 0.0, 0.0],
            [0.1, 0.8, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.8, 0.1, 0.0, 0.0],
        ]
    )
    target = torch.tensor([0, 1, 2])

    # Compute the loss
    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() == pytest.approx(1.25947, rel=1e-3)


def test_triangular_loss_relative():
    loss = TriangularCrossEntropyLoss(num_classes=6)

    input_data = torch.tensor(
        [
            [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    input_data2 = torch.tensor(
        [
            [0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    input_data3 = torch.tensor(
        [
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
        ]
    )

    target = torch.tensor([0])

    # Compute the loss
    output = loss(input_data, target)
    output2 = loss(input_data2, target)
    output3 = loss(input_data3, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    assert output3.item() >= output2.item() > output.item()


def test_triangular_loss_eta():
    input_data = torch.tensor(
        [
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
        ]
    )

    target = torch.tensor([1])

    last_loss = None
    for eta in [0.1, 0.3, 0.5, 0.7, 1.0]:
        loss = TriangularCrossEntropyLoss(num_classes=6, eta=eta)

        # Compute the loss
        output = loss(input_data, target)

        if last_loss is not None:
            assert output.item() < last_loss.item()

        last_loss = output


def test_triangular_loss_alpha2():
    input_data = torch.tensor(
        [
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
        ]
    )

    target = torch.tensor([1])

    last_loss = None
    for alpha2 in [0.01, 0.05, 0.1, 0.15, 0.2]:
        loss = TriangularCrossEntropyLoss(num_classes=6, alpha2=alpha2)

        # Compute the loss
        output = loss(input_data, target)

        if last_loss is not None:
            assert output.item() < last_loss.item()

        last_loss = output


def test_triangular_loss_weights():
    weights = torch.tensor([5.0, 2.0, 1.0, 0.5, 0.1, 0.1])
    loss = TriangularCrossEntropyLoss(num_classes=6, weight=weights)

    input_data = torch.tensor(
        [
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
        ]
    )

    target = torch.tensor([0])
    target2 = torch.tensor([1])
    target3 = torch.tensor([3])

    loss1 = loss(input_data, target)
    loss2 = loss(input_data, target2)
    loss3 = loss(input_data, target3)

    assert loss1.item() > loss2.item() > loss3.item()
