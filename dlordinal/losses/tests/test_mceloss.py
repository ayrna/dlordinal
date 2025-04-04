import pytest
import torch

from dlordinal.losses import MCELoss


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_mceloss_creation(device):
    loss = MCELoss(num_classes=6).to(device)
    loss_logits = MCELoss(num_classes=6, use_logits=True).to(device)
    assert isinstance(loss, MCELoss)
    assert isinstance(loss_logits, MCELoss)


def test_mceloss_basic(device):
    num_classes = 6

    loss = MCELoss(num_classes).to(device)
    loss_logits = MCELoss(num_classes, use_logits=True).to(device)

    input_data = torch.tensor(
        [
            [-2.4079, -2.5133, -2.6187, -2.0652, -3.7299, -5.1068],
            [-2.4079, -2.1725, -2.1459, -3.3318, -3.9624, -4.4700],
            [-2.4079, -1.7924, -2.0101, -4.1030, -3.3445, -4.4812],
        ]
    ).to(device)

    target = torch.tensor([2, 2, 1]).to(device)

    # Compute the loss
    output = loss(torch.nn.functional.softmax(input_data, dim=1), target)
    output_logits = loss_logits(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)
    assert isinstance(output_logits, torch.Tensor)

    assert output.item() > 0
    assert output_logits.item() > 0
    assert output.item() == pytest.approx(output_logits.item(), rel=1e-3)


def test_mceloss_zeroloss(device):
    num_classes = 6

    loss = MCELoss(num_classes).to(device)
    loss_logits = MCELoss(num_classes, use_logits=True).to(device)

    input_data = torch.tensor(
        [
            [0.0, 0.0, 10000.0, 0.0, 0.0, 0.0],
            [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10000.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 10000.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 10000.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 10000.0],
        ]
    ).to(device)

    target = torch.tensor([2, 0, 1, 4, 4, 5]).to(device)

    # Compute the loss
    output = loss(torch.nn.functional.softmax(input_data, dim=1), target)
    output_logits = loss_logits(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)
    assert isinstance(output_logits, torch.Tensor)

    print(output)

    # Verifies that the loss is zero
    assert output.item() == 0.0
    assert output_logits.item() == 0.0


def test_mceloss_exactvalue(device):
    num_classes = 6

    loss = MCELoss(num_classes).to(device)
    loss_logits = MCELoss(num_classes, use_logits=True).to(device)

    input_data = torch.tensor(
        [
            [0.3255, 0.4393, -1.5535, -0.8568, -0.3322, -0.6268],
            [-0.9402, 0.7138, -1.1149, 0.7496, 1.5757, -1.2949],
            [-0.1394, 0.2133, -0.5924, 0.6975, -1.2350, 0.3287],
            [-1.6369, -0.3372, 0.4819, 0.2617, 2.2610, 1.9734],
            [0.3653, -0.7494, 1.4580, 0.6081, 0.8103, 2.0266],
            [0.0591, -0.3830, 1.3315, -0.2456, 0.3857, 0.1374],
            [-0.9824, 0.1123, 0.0179, 0.1617, -1.4736, 0.4105],
            [-0.3583, 0.4595, 2.4428, 0.8525, -1.4174, -0.8660],
            [-0.1506, 0.0283, -0.6797, 0.2403, -0.2998, 0.2174],
            [0.3124, 1.2276, -0.0129, -0.1195, -1.3243, 1.1227],
        ]
    ).to(device)

    target = torch.tensor([1, 2, 4, 3, 2, 0, 5, 3, 1, 3]).to(device)

    # Compute the loss
    output = loss(torch.nn.functional.softmax(input_data, dim=1), target)
    output_logits = loss_logits(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)
    assert isinstance(output_logits, torch.Tensor)

    assert output.item() == pytest.approx(0.1626, rel=1e-3)
    assert output_logits.item() == pytest.approx(0.1626, rel=1e-3)


def test_mceloss_relative(device):
    num_classes = 6

    loss = MCELoss(num_classes).to(device)
    loss_logits = MCELoss(num_classes, use_logits=True).to(device)

    input_data = torch.tensor(
        [
            [0.3255, 0.4393, -1.5535, -0.8568, -0.3322, -0.6268],
            [-0.9402, 0.7138, -1.1149, 0.7496, 1.5757, -1.2949],
            [-0.1394, 0.2133, -0.5924, 0.6975, -1.2350, 0.3287],
            [-1.6369, -0.3372, 0.4819, 0.2617, 2.2610, 1.9734],
            [0.3653, -0.7494, 1.4580, 0.6081, 0.8103, 2.0266],
            [0.0591, -0.3830, 1.3315, -0.2456, 0.3857, 0.1374],
            [-0.9824, 0.1123, 0.0179, 0.1617, -1.4736, 0.4105],
            [-0.3583, 0.4595, 2.4428, 0.8525, -1.4174, -0.8660],
            [-0.1506, 0.0283, -0.6797, 0.2403, -0.2998, 0.2174],
            [0.3124, 1.2276, -0.0129, -0.1195, -1.3243, 1.1227],
        ]
    ).to(device)

    input_data2 = torch.tensor(
        [
            [0.3255, 1.4393, -1.5535, -0.8568, -0.3322, -0.6268],
            [-0.9402, 0.7138, 1.1149, 0.7496, 1.5757, -1.2949],
            [-0.1394, 0.2133, -0.5924, 0.6975, -1.2350, 0.3287],
            [-1.6369, -0.3372, 0.4819, 0.2617, 2.2610, 1.9734],
            [0.3653, -0.7494, 1.4580, 0.6081, 0.8103, 2.0266],
            [0.0591, -0.3830, 1.3315, -0.2456, 0.3857, 0.1374],
            [-0.9824, 0.1123, 0.0179, 0.1617, -1.4736, 0.4105],
            [-0.3583, 0.4595, 2.4428, 0.8525, -1.4174, -0.8660],
            [-0.1506, 0.0283, -0.6797, 0.2403, -0.2998, 0.2174],
            [0.3124, 1.2276, -0.0129, -0.1195, -1.3243, 1.1227],
        ]
    ).to(device)

    target = torch.tensor([1, 2, 4, 3, 2, 0, 5, 3, 1, 3]).to(device)

    # Compute the loss
    output = loss(torch.nn.functional.softmax(input_data, dim=1), target)
    output2 = loss(torch.nn.functional.softmax(input_data2, dim=1), target)
    output_logits = loss_logits(input_data, target)
    output2_logits = loss_logits(input_data2, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)
    assert isinstance(output2, torch.Tensor)
    assert isinstance(output_logits, torch.Tensor)
    assert isinstance(output2_logits, torch.Tensor)

    assert output.item() > output2.item()
    assert output_logits.item() > output2_logits.item()


def test_mceloss_onlyoneclassperfect(device):
    num_classes = 6

    loss = MCELoss(num_classes).to(device)
    loss_logits = MCELoss(num_classes, use_logits=True).to(device)

    input_data = torch.tensor(
        [
            [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10000.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 10000.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 10000.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 10000.0],
            [0.0, 0.0, 0.0, 0.0, 10000.0, 0.0],
        ]
    ).to(device)

    target = torch.tensor([0, 0, 0, 1, 2, 3, 4, 5]).to(device)

    # Compute the loss
    output = loss(torch.nn.functional.softmax(input_data, dim=1), target)
    output_logits = loss_logits(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)
    assert isinstance(output_logits, torch.Tensor)

    assert output.item() == pytest.approx(0.2083, rel=1e-3)
    assert output_logits.item() == pytest.approx(0.2083, rel=1e-3)


def test_mceloss_perclassmse(device):
    num_classes = 6

    loss = MCELoss(num_classes).to(device)
    loss_logits = MCELoss(num_classes, use_logits=True).to(device)

    input_data = torch.tensor(
        [
            [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10000.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 10000.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 10000.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 10000.0],
            [0.0, 0.0, 0.0, 0.0, 10000.0, 0.0],
        ]
    ).to(device)

    input_data2 = torch.tensor(
        [
            [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10000.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10000.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 10000.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 10000.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 10000.0],
            [0.0, 0.0, 0.0, 0.0, 10000.0, 0.0],
        ]
    ).to(device)

    input_data3 = torch.tensor(
        [
            [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 10000.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 10000.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 10000.0],
            [0.0, 0.0, 0.0, 0.0, 10000.0, 0.0],
        ]
    ).to(device)

    target = torch.tensor([0, 0, 0, 1, 2, 3, 4, 5]).to(device)
    target = torch.nn.functional.one_hot(target)

    # Compute the loss
    per_class_mse = loss.compute_per_class_mse(
        torch.nn.functional.softmax(input_data, dim=1), target
    )
    per_class_mse2 = loss.compute_per_class_mse(
        torch.nn.functional.softmax(input_data2, dim=1), target
    )
    per_class_mse3 = loss.compute_per_class_mse(
        torch.nn.functional.softmax(input_data3, dim=1), target
    )

    per_class_mse_logits = loss_logits.compute_per_class_mse(input_data, target)
    per_class_mse2_logits = loss_logits.compute_per_class_mse(input_data2, target)
    per_class_mse3_logits = loss_logits.compute_per_class_mse(input_data3, target)

    # Verifies that the output is a tensor
    assert isinstance(per_class_mse, torch.Tensor)
    assert isinstance(per_class_mse2, torch.Tensor)
    assert isinstance(per_class_mse3, torch.Tensor)
    assert isinstance(per_class_mse_logits, torch.Tensor)
    assert isinstance(per_class_mse2_logits, torch.Tensor)
    assert isinstance(per_class_mse3_logits, torch.Tensor)

    # Check that the MSE of the first class should increase when there are errors
    # in that class. It should increase in the same way with FP or FN.
    assert per_class_mse2[0] > per_class_mse[0]
    assert per_class_mse3[0] > per_class_mse[0]
    assert per_class_mse3[0] == per_class_mse2[0]

    assert per_class_mse2_logits[0] > per_class_mse_logits[0]
    assert per_class_mse3_logits[0] > per_class_mse_logits[0]
    assert per_class_mse3_logits[0] == per_class_mse2_logits[0]

    # Check that the error in the rest of the classes should remain constant.
    # Classes 0 and 1 are not checked cause they were modified.
    assert (per_class_mse2[2:] == per_class_mse[2:]).all()
    assert (per_class_mse3[2:] == per_class_mse[2:]).all()

    assert (per_class_mse2_logits[2:] == per_class_mse_logits[2:]).all()
    assert (per_class_mse3_logits[2:] == per_class_mse_logits[2:]).all()


def test_mceloss_weights(device):
    num_classes = 6
    weight = torch.tensor(
        [1.60843373, 0.55394191, 1.02692308, 0.78070175, 1.12184874, 2.34210526],
        dtype=torch.float,
    ).to(device)

    loss = MCELoss(num_classes).to(device)
    loss_weighted = MCELoss(num_classes, weight=weight).to(device)
    loss_logits = MCELoss(num_classes, use_logits=True).to(device)
    loss_logits_weighted = MCELoss(num_classes, weight=weight, use_logits=True).to(
        device
    )

    # Input data without errors
    input_data = torch.tensor(
        [
            [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10000.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10000.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10000.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1000.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 10000.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 10000.0],
        ]
    ).to(device)

    # Input data with error in classes 0 and 1
    input_data2 = torch.tensor(
        [
            [0.0, 10000.0, 0.0, 0.0, 0.0, 0.0],
            [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10000.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10000.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10000.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1000.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 10000.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 10000.0],
        ]
    ).to(device)

    # Input data with error in classes 0 and 2
    input_data3 = torch.tensor(
        [
            [0.0, 0.0, 10000.0, 0.0, 0.0, 0.0],
            [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10000.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10000.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10000.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1000.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 10000.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 10000.0],
        ]
    ).to(device)

    target = torch.tensor([0, 0, 1, 2, 2, 3, 4, 5]).to(device)

    # Compute the loss
    output = loss(torch.nn.functional.softmax(input_data, dim=1), target)
    output2 = loss(torch.nn.functional.softmax(input_data2, dim=1), target)
    output3 = loss(torch.nn.functional.softmax(input_data3, dim=1), target)

    output_logits = loss_logits(input_data, target)
    output2_logits = loss_logits(input_data2, target)
    output3_logits = loss_logits(input_data3, target)

    output_weighted = loss_weighted(
        torch.nn.functional.softmax(input_data, dim=1), target
    )
    output2_weighted = loss_weighted(
        torch.nn.functional.softmax(input_data2, dim=1), target
    )
    output3_weighted = loss_weighted(
        torch.nn.functional.softmax(input_data3, dim=1), target
    )

    output_logits_weighted = loss_logits_weighted(input_data, target)
    output2_logits_weighted = loss_logits_weighted(input_data2, target)
    output3_logits_weighted = loss_logits_weighted(input_data3, target)

    # Check that the error is the same in weighted and non-weighted loss when there
    # are no errors
    assert output_weighted.item() == output.item()
    assert output_logits_weighted.item() == output_logits.item()

    # Check that the weighted loss is higher when the weights of the classes with
    # errors is greater than 1.
    assert output2_weighted.item() > output2.item()
    assert output3_weighted.item() > output3.item()
    assert output2_logits_weighted.item() > output2_logits.item()
    assert output3_logits_weighted.item() > output3_logits.item()
