import pytest
import torch

from dlordinal.losses import GeometricCrossEntropyLoss


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_geometric_loss_creation(device):
    loss = GeometricCrossEntropyLoss(num_classes=5).to(device)
    assert isinstance(loss, GeometricCrossEntropyLoss)


def test_geometric_loss_basic(device):
    loss = GeometricCrossEntropyLoss(num_classes=6).to(device)

    input_data = torch.tensor(
        [
            [-1.6488, -2.5838, -2.8312, -1.9495, -2.4759, -3.4682],
            [-1.7872, -3.9560, -6.2586, -8.3967, -7.9779, -8.0079],
            [-2.4078, -2.5133, -2.5584, -1.7485, -2.3675, -2.6099],
        ]
    ).to(device)
    target = torch.tensor([4, 0, 5]).to(device)

    # Compute the loss
    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() > 0


def test_geometric_loss_relative(device):
    loss = GeometricCrossEntropyLoss(num_classes=6).to(device)

    input_data = torch.tensor(
        [
            [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    ).to(device)

    input_data2 = torch.tensor(
        [
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
        ]
    ).to(device)

    input_data3 = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 100.0, 0.0],
        ]
    ).to(device)

    target = torch.tensor([0]).to(device)

    # Compute the loss
    output = loss(input_data, target)
    output2 = loss(input_data2, target)
    output3 = loss(input_data3, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    assert output3.item() > output2.item() > output.item()
