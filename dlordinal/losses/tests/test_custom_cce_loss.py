import pytest
import torch

from dlordinal.losses import CustomTargetsCrossEntropyLoss

# TODO: remove in 3.0.0
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# Auxiliar function to get a test class
def create_loss_class():
    cls_probs = torch.tensor([[0.6, 0.2, 0.2], [0.4, 0.5, 0.1], [0.1, 0.2, 0.7]])
    return CustomTargetsCrossEntropyLoss(cls_probs)


# Test the creation of the CustomTargetsCrossEntropyLoss class
def test_custom_loss_creation():
    loss = create_loss_class()
    assert isinstance(loss, CustomTargetsCrossEntropyLoss)


# Test the calculation of the loss
def test_custom_loss_forward():
    loss = create_loss_class()

    input_data = torch.tensor([[0.1, 0.5, 0.4], [0.7, 0.2, 0.1]])
    target = torch.tensor([0, 1])

    # Compute the loss
    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() > 0


def test_custom_targets_cross_entropy_loss_label_smoothing():
    cls_probs = torch.eye(3)
    loss_fn = CustomTargetsCrossEntropyLoss(cls_probs, eta=0.0, label_smoothing=0.3)
    assert loss_fn.base_loss.label_smoothing == 0.3
    input = torch.randn(4, 3)
    target = torch.tensor([0, 1, 2, 0])
    expected = torch.nn.CrossEntropyLoss(label_smoothing=0.3)(input, target)
    assert loss_fn(input, target).item() == pytest.approx(expected.item())
