import torch

from ..triangular_loss import TriangularCrossEntropyLoss


def test_triangular_loss_creation():
    loss = TriangularCrossEntropyLoss()
    assert isinstance(loss, TriangularCrossEntropyLoss)


def test_triangular_loss_output():
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
