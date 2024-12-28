import torch

from dlordinal.losses import GeometricCrossEntropyLoss


def test_geometric_loss_creation():
    loss = GeometricCrossEntropyLoss(num_classes=5)
    assert isinstance(loss, GeometricCrossEntropyLoss)


def test_geometric_loss():
    loss = GeometricCrossEntropyLoss(num_classes=6)

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
