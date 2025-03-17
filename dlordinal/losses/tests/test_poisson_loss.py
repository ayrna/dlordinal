import pytest
import torch

from dlordinal.losses import PoissonCrossEntropyLoss


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_poisson_loss_creation(device):
    loss = PoissonCrossEntropyLoss(num_classes=5).to(device)
    assert isinstance(loss, PoissonCrossEntropyLoss)


def test_poisson_loss_basic(device):
    loss = PoissonCrossEntropyLoss(num_classes=6).to(device)

    input_data = torch.tensor(
        [
            [-2.3384, -2.5224, -2.6278, -2.7332, -2.8385, -1.9286],
            [-2.4079, -2.5133, -2.6187, -1.7466, -2.2076, -2.9389],
            [-2.4079, -2.5133, -2.6187, -2.7052, -2.7559, -1.9257],
        ]
    ).to(device)
    target = torch.tensor([5, 3, 5]).to(device)

    # Compute the loss
    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() > 0


def test_poisson_loss_exactvalue(device):
    loss = PoissonCrossEntropyLoss(num_classes=6).to(device)

    input_data = torch.tensor(
        [
            [0.8, 0.1, 0.1, 0.0, 0.0, 0.0],
            [0.1, 0.8, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.8, 0.1, 0.0, 0.0],
        ]
    ).to(device)
    target = torch.tensor([0, 1, 2]).to(device)

    # Compute the loss
    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() == pytest.approx(1.8203, rel=1e-3)


def test_poisson_loss_relative(device):
    loss = PoissonCrossEntropyLoss(num_classes=6).to(device)

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


def test_poisson_loss_eta(device):
    input_data = torch.tensor(
        [
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
        ]
    ).to(device)

    target = torch.tensor([0]).to(device)

    last_loss = None
    for eta in [0.1, 0.3, 0.5, 0.7, 1.0]:
        loss = PoissonCrossEntropyLoss(num_classes=6, eta=eta).to(device)

        # Compute the loss
        output = loss(input_data, target)

        if last_loss is not None:
            assert output.item() < last_loss.item()

        last_loss = output


def test_poisson_loss_weights(device):
    weights = torch.tensor([5.0, 2.0, 1.0, 0.5, 0.1, 0.1]).to(device)
    loss = PoissonCrossEntropyLoss(num_classes=6, weight=weights).to(device)

    input_data = torch.tensor(
        [
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
        ]
    ).to(device)

    target = torch.tensor([0]).to(device)
    target2 = torch.tensor([1]).to(device)
    target3 = torch.tensor([3]).to(device)

    loss1 = loss(input_data, target)
    loss2 = loss(input_data, target2)
    loss3 = loss(input_data, target3)

    assert loss1.item() > loss2.item() > loss3.item()
