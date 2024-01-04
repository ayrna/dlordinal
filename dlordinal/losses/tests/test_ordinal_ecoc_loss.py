import torch
from torch import cuda

from ..ordinal_ecoc_distance_loss import OrdinalEcocDistanceLoss


def test_ordinal_ecoc_distance_loss_creation():
    device = "cuda" if cuda.is_available() else "cpu"

    loss = OrdinalEcocDistanceLoss(num_classes=6, device=device)
    assert isinstance(loss, OrdinalEcocDistanceLoss)


def test_ordinal_ecoc_distance_loss_output():
    device = "cuda" if cuda.is_available() else "cpu"

    num_classes = 6
    loss = OrdinalEcocDistanceLoss(num_classes, device=device)

    input_data = torch.tensor(
        [
            [0.4492, 0.5579, 0.4470, 0.4841, 0.3665],
            [0.4532, 0.5256, 0.4102, 0.5699, 0.3539],
            [0.5053, 0.4867, 0.3301, 0.5264, 0.3328],
            [0.4698, 0.4144, 0.3409, 0.5315, 0.3169],
            [0.4352, 0.5462, 0.3624, 0.5480, 0.4264],
        ]
    ).to(device)

    target = torch.tensor([3, 1, 1, 1, 1]).to(device)

    # Compute the loss
    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() > 0
