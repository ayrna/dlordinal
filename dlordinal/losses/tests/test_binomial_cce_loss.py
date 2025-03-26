import pytest
import torch

from dlordinal.losses import BinomialCrossEntropyLoss


def test_binomial_loss_creation():
    loss = BinomialCrossEntropyLoss(num_classes=5)
    assert isinstance(loss, BinomialCrossEntropyLoss)


def test_binomial_loss_basic():
    loss = BinomialCrossEntropyLoss(num_classes=6)

    input_data = torch.tensor(
        [
            [-2.4079, -2.5133, -1.9160, -1.9258, -2.3771, -3.2150],
            [-2.4079, -2.5133, -2.6187, -1.7665, -2.1753, -2.9375],
            [-2.4079, -2.5133, -2.6187, -2.7240, -2.0245, -2.1541],
        ]
    )

    target = torch.tensor([1, 3, 4])

    # Compute the loss
    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() > 0


def test_binomial_loss_exactvalue():
    loss = BinomialCrossEntropyLoss(num_classes=6)

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
    assert output.item() == pytest.approx(1.60699, rel=1e-3)


def test_binomial_loss_relative():
    loss = BinomialCrossEntropyLoss(num_classes=6)

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

    input_data3 = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 100.0, 0.0],
        ]
    )

    target = torch.tensor([0])

    # Compute the loss
    output = loss(input_data, target)
    output2 = loss(input_data2, target)
    output3 = loss(input_data3, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    assert output3.item() > output2.item() > output.item()


def test_binomial_loss_eta():
    input_data = torch.tensor(
        [
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
        ]
    )

    target = torch.tensor([0])

    last_loss = None
    for eta in [0.1, 0.3, 0.5, 0.7, 1.0]:
        loss = BinomialCrossEntropyLoss(num_classes=6, eta=eta)

        # Compute the loss
        output = loss(input_data, target)

        if last_loss is not None:
            assert output.item() < last_loss.item()

        last_loss = output


def test_binomial_loss_weights():
    weights = torch.tensor([5.0, 2.0, 1.0, 0.5, 0.1, 0.1])
    loss = BinomialCrossEntropyLoss(num_classes=6, weight=weights)

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
