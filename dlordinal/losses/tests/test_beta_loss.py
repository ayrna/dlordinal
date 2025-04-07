import pytest
import torch
from torch.nn import CrossEntropyLoss

from dlordinal.losses import BetaLoss


@pytest.fixture
def device():
    d = "cpu"

    if torch.cuda.is_available():
        d = "cuda"

    return d


def test_beta_loss_creation(device):
    base_loss = CrossEntropyLoss().to(device)
    loss = BetaLoss(base_loss=base_loss, num_classes=5).to(device)
    assert isinstance(loss, BetaLoss)


def test_beta_loss_basic(device):
    base_loss = CrossEntropyLoss().to(device)
    loss = BetaLoss(base_loss=base_loss, num_classes=6).to(device)

    input_data = torch.tensor(
        [
            [0.4965, 0.5200, 0.2156, 0.9261, -0.6116, 1.0949],
            [-0.4715, -0.7595, 1.1330, 0.7932, 0.0749, 1.2884],
            [0.8929, 0.5330, 0.0984, 0.3900, -0.7238, 0.4939],
        ]
    ).to(device)
    target = torch.tensor([5, 3, 1]).to(device)

    # Compute the loss
    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() > 0


def test_beta_loss_exactvalue(device):
    base_loss = CrossEntropyLoss().to(device)
    loss = BetaLoss(base_loss=base_loss, num_classes=6).to(device)

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
    assert output.item() == pytest.approx(1.3925, rel=1e-3)


def test_beta_loss_relative(device):
    base_loss = CrossEntropyLoss().to(device)
    loss = BetaLoss(base_loss=base_loss, num_classes=6).to(device)

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


def test_beta_loss_eta(device):
    input_data = torch.tensor(
        [
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
        ]
    ).to(device)

    target = torch.tensor([0]).to(device)

    last_loss = None
    for eta in [0.1, 0.3, 0.5, 0.7, 1.0]:
        base_loss = CrossEntropyLoss().to(device)
        loss = BetaLoss(base_loss=base_loss, num_classes=6, eta=eta).to(device)

        # Compute the loss
        output = loss(input_data, target)

        if last_loss is not None:
            assert output.item() < last_loss.item()

        last_loss = output


def test_beta_loss_weights(device):
    weights = torch.tensor([5.0, 2.0, 1.0, 0.5, 0.1, 0.1])
    base_loss = CrossEntropyLoss(weight=weights).to(device)
    loss = BetaLoss(base_loss=base_loss, num_classes=6).to(device)

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


@pytest.mark.parametrize("J", [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
def test_beta_loss_custom_params_set(device, J):
    from dlordinal.soft_labelling.beta_distribution import _beta_params_sets

    base_loss = CrossEntropyLoss().to(device)

    loss_standard = BetaLoss(
        base_loss=base_loss, num_classes=J, params_set="standard"
    ).to(device)

    params = _beta_params_sets["standard"]
    loss_custom = BetaLoss(
        base_loss=base_loss,
        num_classes=J,
        params_set=params,
    ).to(device)

    input_data = torch.rand(20, J).to(device)
    target = torch.randint(0, J, (20,)).to(device)

    output_standard = loss_standard(input_data, target)
    output_custom = loss_custom(input_data, target)

    assert output_standard.item() == output_custom.item()
