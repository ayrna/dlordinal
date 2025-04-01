import pytest
import torch

from dlordinal.losses import BetaCrossEntropyLoss


def test_beta_loss_creation():
    loss = BetaCrossEntropyLoss(num_classes=5)
    assert isinstance(loss, BetaCrossEntropyLoss)


def test_beta_loss_basic():
    loss = BetaCrossEntropyLoss(num_classes=6)

    input_data = torch.tensor(
        [
            [0.4965, 0.5200, 0.2156, 0.9261, -0.6116, 1.0949],
            [-0.4715, -0.7595, 1.1330, 0.7932, 0.0749, 1.2884],
            [0.8929, 0.5330, 0.0984, 0.3900, -0.7238, 0.4939],
        ]
    )
    target = torch.tensor([5, 3, 1])

    # Compute the loss
    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() > 0


def test_beta_loss_exactvalue():
    loss = BetaCrossEntropyLoss(num_classes=6)

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
    assert output.item() == pytest.approx(1.3925, rel=1e-3)


def test_beta_loss_relative():
    loss = BetaCrossEntropyLoss(num_classes=6)

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


def test_beta_loss_eta():
    input_data = torch.tensor(
        [
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
        ]
    )

    target = torch.tensor([0])

    last_loss = None
    for eta in [0.1, 0.3, 0.5, 0.7, 1.0]:
        loss = BetaCrossEntropyLoss(num_classes=6, eta=eta)

        # Compute the loss
        output = loss(input_data, target)

        if last_loss is not None:
            assert output.item() < last_loss.item()

        last_loss = output


def test_beta_loss_weights():
    weights = torch.tensor([5.0, 2.0, 1.0, 0.5, 0.1, 0.1])
    loss = BetaCrossEntropyLoss(num_classes=6, weight=weights)

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
