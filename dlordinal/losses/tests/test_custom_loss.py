import torch

from dlordinal.losses import CustomTargetsCrossEntropyLoss


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
