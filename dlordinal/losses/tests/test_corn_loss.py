import pytest
import torch

from dlordinal.losses import CORNLoss


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_cornloss_creation(device):
    loss = CORNLoss(num_classes=6).to(device)
    assert isinstance(loss, CORNLoss)


def test_cornloss_basic(device):
    num_classes = 6

    loss_logits = CORNLoss(num_classes).to(device)

    input_data = torch.tensor(
        [
            [-2.4079, -2.5133, -2.6187, -2.0652, -3.7299],
            [-2.4079, -2.1725, -2.1459, -3.3318, -3.9624],
            [-2.4079, -1.7924, -2.0101, -4.1030, -3.3445],
        ]
    ).to(device)

    target = torch.tensor([2, 2, 1]).to(device)

    # Compute the loss
    output_logits = loss_logits(input_data, target)

    assert isinstance(output_logits, torch.Tensor)
    assert output_logits.item() > 0


def test_cornloss_zeroloss(device):
    num_classes = 3

    loss = CORNLoss(num_classes).to(device)

    zero = -1e3
    one = 1e3

    input_data = torch.tensor(
        [
            [zero, one],
            [one, zero],
            [one, one],
        ]
    ).to(device)

    target = torch.tensor([0, 1, 2]).to(device)

    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Check that the loss is zero
    assert output.item() == pytest.approx(0.0, rel=1e-6)
