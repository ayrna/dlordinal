import torch

from ..beta_loss import BetaCrossEntropyLoss


def test_beta_loss_creation():
    loss = BetaCrossEntropyLoss()
    assert isinstance(loss, BetaCrossEntropyLoss)


def test_beta_loss_output():
    loss = BetaCrossEntropyLoss(num_classes=6)

    input_data = torch.tensor(
        [
            [0.4965, 0.5200, 0.2156, 0.9261, -0.6116, 1.0949],
            [-0.4715, -0.7595, 1.1330, 0.7932, 0.0749, 1.2884],
            [0.8929, 0.5330, 0.0984, 0.3900, -0.7238, 0.4939],
        ]
    )
    target = torch.tensor([5, 3, 1])

    # Compute the loss
    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() > 0
