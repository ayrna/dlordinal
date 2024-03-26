import pytest
import torch

from dlordinal.losses import ExponentialRegularisedCrossEntropyLoss


def test_exponential_loss_creation():
    loss = ExponentialRegularisedCrossEntropyLoss(num_classes=5)
    assert isinstance(loss, ExponentialRegularisedCrossEntropyLoss)


def test_exponential_loss_basic():
    loss = ExponentialRegularisedCrossEntropyLoss(num_classes=6)

    input_data = torch.tensor(
        [
            [-1.8020, -2.6416, -2.7470, -2.3354, -2.2209, -3.2640],
            [-2.4079, -2.5133, -2.6187, -2.7240, -1.8659, -2.8370],
            [-2.4079, -4.7105, -7.0131, -9.3157, -9.4211, -8.5060],
        ]
    )
    target = torch.tensor([2, 5, 0])

    # Compute the loss
    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() > 0


def test_exponential_loss_exactvalue():
    for p, expected in [
        (1.0, 1.53492),
        (1.2, 1.51328),
        (1.4, 1.49786),
        (1.6, 1.48705),
        (1.8, 1.47952),
        (2.0, 1.47439),
    ]:
        loss = ExponentialRegularisedCrossEntropyLoss(num_classes=6, p=p)

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
        assert output.item() == pytest.approx(expected, rel=1e-3)


def test_exponential_loss_relative():
    loss = ExponentialRegularisedCrossEntropyLoss(num_classes=6)

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


def test_exponential_loss_eta():
    input_data = torch.tensor(
        [
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
        ]
    )

    target = torch.tensor([0])

    last_loss = None
    for eta in [0.1, 0.3, 0.5, 0.7, 1.0]:
        loss = ExponentialRegularisedCrossEntropyLoss(num_classes=6, eta=eta)

        # Compute the loss
        output = loss(input_data, target)

        if last_loss is not None:
            assert output.item() < last_loss.item()

        last_loss = output


def test_exponential_loss_p():
    input_data = torch.tensor(
        [
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
        ]
    )

    target = torch.tensor([0])

    last_loss = None
    for p in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
        loss = ExponentialRegularisedCrossEntropyLoss(num_classes=6, p=p)

        # Compute the loss
        output = loss(input_data, target)

        if last_loss is not None:
            assert output.item() > last_loss.item()

        last_loss = output


def test_exponential_loss_weights():
    weights = torch.tensor([5.0, 2.0, 1.0, 0.5, 0.1, 0.1])
    loss = ExponentialRegularisedCrossEntropyLoss(num_classes=6, weight=weights)

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
