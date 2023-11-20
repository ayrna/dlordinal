import pytest
import torch

from ..losses import BinomialCrossEntropyLoss


def test_binomial_loss_creation():
    loss = BinomialCrossEntropyLoss()
    assert isinstance(loss, BinomialCrossEntropyLoss)


def test_binomial_loss_output():
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


if __name__ == "__main__":
    test_binomial_loss_creation()
    test_binomial_loss_output()
