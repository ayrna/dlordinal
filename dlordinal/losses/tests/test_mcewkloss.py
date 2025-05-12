import pytest
import torch

from dlordinal.losses import MCEAndWKLoss, MCELoss, WKLoss


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_mcewkloss_creation(device):
    loss = MCEAndWKLoss(num_classes=6).to(device)
    loss_logits = MCEAndWKLoss(num_classes=6, use_logits=True).to(device)
    assert isinstance(loss, MCEAndWKLoss)
    assert isinstance(loss_logits, MCEAndWKLoss)


def test_mcewkloss_basic(device):
    num_classes = 6
    C = 0.5
    penalization_type = "quadratic"

    loss = MCEAndWKLoss(num_classes, C=C, wk_penalization_type=penalization_type).to(
        device
    )
    loss_logits = MCEAndWKLoss(
        num_classes, C=C, wk_penalization_type=penalization_type, use_logits=True
    ).to(device)

    input_data = torch.tensor(
        [
            [-2.4079, -2.5133, -2.6187, -2.0652, -3.7299, -5.1068],
            [-2.4079, -2.1725, -2.1459, -3.3318, -3.9624, -4.4700],
            [-2.4079, -1.7924, -2.0101, -4.1030, -3.3445, -4.4812],
        ]
    ).to(device)

    target = torch.tensor([2, 2, 1]).to(device)

    # Compute the loss
    output = loss(target, torch.nn.functional.softmax(input_data, dim=1))
    output_logits = loss_logits(target, input_data)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)
    assert isinstance(output_logits, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() > 0
    assert output_logits.item() > 0
    assert output.item() == pytest.approx(output_logits.item())


def test_mcewkloss_exactvalue(device):
    num_classes = 6
    C = 0.5
    penalization_type = "quadratic"

    loss = MCEAndWKLoss(num_classes, C=C, wk_penalization_type=penalization_type).to(
        device
    )
    loss_logits = MCEAndWKLoss(
        num_classes, C=C, wk_penalization_type=penalization_type, use_logits=True
    ).to(device)

    input_data = torch.tensor(
        [
            [0.4771, 1.7111, -0.5724, 0.2167, -0.6971, 0.0899],
            [-0.3333, 2.2221, 1.4642, 0.5251, -1.9387, -0.1995],
            [-1.2130, -0.5895, -1.0779, -0.4865, -0.3104, -0.2203],
            [1.9028, -0.0311, 1.0128, -0.3319, -1.9604, -1.0178],
            [-0.3340, 0.1268, 1.8875, -0.2423, 0.8052, 0.5424],
            [-1.6172, -0.9965, 0.3532, -0.0951, -1.3067, -0.0890],
            [-0.4112, -0.6456, -0.7137, -0.6413, -0.9140, -0.3890],
            [0.6984, 2.2330, 0.5405, 1.2638, 0.0382, 1.4028],
            [-0.3806, -2.0569, -0.8110, -2.1876, -0.0553, 0.0071],
            [1.0932, 1.0941, 0.5114, -0.3906, 0.3693, 1.7794],
        ]
    ).to(device)

    target = torch.tensor([0, 1, 2, 3, 4, 5, 4, 3, 2, 1]).to(device)

    # Compute the loss
    output = loss(target, torch.nn.functional.softmax(input_data, dim=1))
    output_logits = loss_logits(target, input_data)

    assert output.item() == pytest.approx(0.5404, rel=1e-3)
    assert output_logits.item() == pytest.approx(0.5404, rel=1e-3)


def test_mcewkloss_weights(device):
    num_classes = 6
    C = 0.5
    penalization_type = "quadratic"
    weight = torch.tensor(
        [1.60843373, 0.55394191, 1.02692308, 0.78070175, 1.12184874, 2.34210526],
        dtype=torch.float,
    ).to(device)

    loss = MCEAndWKLoss(num_classes, C=C, wk_penalization_type=penalization_type).to(
        device
    )
    loss_weighted = MCEAndWKLoss(
        num_classes, C=C, wk_penalization_type=penalization_type, weight=weight
    ).to(device)
    loss_logits = MCEAndWKLoss(
        num_classes,
        C=C,
        wk_penalization_type=penalization_type,
        use_logits=True,
    ).to(device)
    loss_logits_weighted = MCEAndWKLoss(
        num_classes,
        C=C,
        wk_penalization_type=penalization_type,
        weight=weight,
        use_logits=True,
    ).to(device)

    input_data = torch.tensor(
        [
            [-0.2031, 1.4755, 0.0284, -0.2866, 0.1422, 0.3226],
            [-0.2673, 0.0656, 0.7692, 1.4546, -0.3020, 0.3431],
            [0.8162, 1.3829, 0.1576, -0.1615, 1.2485, 0.2667],
        ]
    ).to(device)

    # Error in classes 1 and 3 (pattern from class 1 classified as 3)
    target = torch.tensor([3, 3, 1]).to(device)

    # Compute the loss
    output = loss(target, torch.nn.functional.softmax(input_data, dim=1))
    output_weighted = loss_weighted(
        target, torch.nn.functional.softmax(input_data, dim=1)
    )
    output_logits = loss_logits(target, input_data)
    output_logits_weighted = loss_logits_weighted(target, input_data)

    # The error should be lower when using the weighted version as the weight
    # of classes 1 and 3 is less than 1
    assert output.item() > output_weighted.item()
    assert output_logits.item() > output_logits_weighted.item()

    # Error in classes 1 and 5 (pattern from class 1 classified as 5)
    target = torch.tensor([5, 3, 1]).to(device)

    # Compute the loss
    output = loss(target, torch.nn.functional.softmax(input_data, dim=1))
    output_weighted = loss_weighted(
        target, torch.nn.functional.softmax(input_data, dim=1)
    )
    output_logits = loss_logits(target, input_data)
    output_logits_weighted = loss_logits_weighted(target, input_data)

    # The error should be higher when using the weighted version as the weight
    # of class 5 is way higher than 1
    assert output.item() < output_weighted.item()
    assert output_logits.item() < output_logits_weighted.item()


def test_mcewkloss_combination(device):
    num_classes = 6
    C = 0.5
    penalization_type = "quadratic"

    loss = MCEAndWKLoss(num_classes, C=C, wk_penalization_type=penalization_type).to(
        device
    )
    mce_loss = MCELoss(num_classes).to(device)
    wk_loss = WKLoss(num_classes, penalization_type=penalization_type).to(device)

    loss_logits = MCEAndWKLoss(
        num_classes, C=C, wk_penalization_type=penalization_type, use_logits=True
    ).to(device)
    mce_loss_logits = MCELoss(num_classes, use_logits=True).to(device)
    wk_loss_logits = WKLoss(
        num_classes, penalization_type=penalization_type, use_logits=True
    ).to(device)

    input_data = torch.tensor(
        [
            [-0.2031, 1.4755, 0.0284, -0.2866, 0.1422, 0.3226],
            [-0.2673, 0.0656, 0.7692, 1.4546, -0.3020, 0.3431],
            [0.8162, 1.3829, 0.1576, -0.1615, 1.2485, 0.2667],
        ]
    ).to(device)

    # Error in classes 1 and 3 (pattern from class 1 classified as 3)
    target = torch.tensor([3, 3, 1]).to(device)

    # Compute the loss
    output = loss(target, torch.nn.functional.softmax(input_data, dim=1))
    output_mce = mce_loss(torch.nn.functional.softmax(input_data, dim=1), target)
    output_wk = wk_loss(torch.nn.functional.softmax(input_data, dim=1), target)
    output_combined = C * output_mce + (1 - C) * output_wk

    output_logits = loss_logits(target, input_data)
    output_mce_logits = mce_loss_logits(input_data, target)
    output_wk_logits = wk_loss_logits(input_data, target)
    output_combined_logits = C * output_mce_logits + (1 - C) * output_wk_logits

    assert output.item() == pytest.approx(output_combined.item())
    assert output_logits.item() == pytest.approx(output_combined_logits.item())
    assert output.item() == pytest.approx(output_combined_logits.item())
    assert output_logits.item() == pytest.approx(output_combined.item())
