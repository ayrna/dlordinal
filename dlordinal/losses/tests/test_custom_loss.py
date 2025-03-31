import pytest
import torch
from torch.nn import CrossEntropyLoss

from dlordinal.losses import CustomTargetsLoss


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# Auxiliar function to get a test class
def create_loss_class(device):
    cls_probs = torch.tensor([[0.6, 0.2, 0.2], [0.4, 0.5, 0.1], [0.1, 0.2, 0.7]]).to(
        device
    )
    base_loss = CrossEntropyLoss().to(device)
    loss = CustomTargetsLoss(base_loss=base_loss, cls_probs=cls_probs).to(device)
    return loss


# Test the creation of the CustomTargetsCrossEntropyLoss class
def test_custom_loss_creation(device):
    loss = create_loss_class(device)
    assert isinstance(loss, CustomTargetsLoss)


# Test the calculation of the loss
def test_custom_loss_forward(device):
    loss = create_loss_class(device)

    input_data = torch.tensor([[0.1, 0.5, 0.4], [0.7, 0.2, 0.1]]).to(device)
    target = torch.tensor([0, 1]).to(device)

    # Compute the loss
    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() > 0
