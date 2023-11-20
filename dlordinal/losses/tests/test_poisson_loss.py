import pytest
import torch

from ..losses import PoissonCrossEntropyLoss


def test_poisson_loss_creation():
    loss = PoissonCrossEntropyLoss()
    assert isinstance(loss, PoissonCrossEntropyLoss)


def test_poisson_loss_output():
    loss = PoissonCrossEntropyLoss(num_classes=6)

    input_data = torch.tensor(
        [
            [-2.3384, -2.5224, -2.6278, -2.7332, -2.8385, -1.9286],
            [-2.4079, -2.5133, -2.6187, -1.7466, -2.2076, -2.9389],
            [-2.4079, -2.5133, -2.6187, -2.7052, -2.7559, -1.9257],
        ]
    )
    target = torch.tensor([5, 3, 5])

    # Compute the loss
    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() > 0


if __name__ == "__main__":
    test_poisson_loss_creation()
    test_poisson_loss_output()
