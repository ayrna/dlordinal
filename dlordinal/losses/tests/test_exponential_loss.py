import torch

from ..exponential_loss import ExponentialRegularisedCrossEntropyLoss


def test_exponential_loss_creation():
    loss = ExponentialRegularisedCrossEntropyLoss()
    assert isinstance(loss, ExponentialRegularisedCrossEntropyLoss)


def test_exponential_loss_output():
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
