import pytest
import torch

from dlordinal.losses import WKLoss


def test_wkloss_creation():
    loss = WKLoss(num_classes=6)
    assert isinstance(loss, WKLoss)


def test_wkloss_basic():
    num_classes = 6
    penalization_type = "quadratic"

    loss = WKLoss(num_classes, penalization_type)

    input_data = torch.tensor(
        [
            [-2.4079, -2.5133, -2.6187, -2.0652, -3.7299, -5.1068],
            [-2.4079, -2.1725, -2.1459, -3.3318, -3.9624, -4.4700],
            [-2.4079, -1.7924, -2.0101, -4.1030, -3.3445, -4.4812],
        ]
    )

    target = torch.tensor([2, 2, 1])

    # Compute the loss
    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() > 0


def test_wkloss_custom_weight():
    num_classes = 6
    penalization_type = "quadratic"
    weight = torch.tensor(
        [1.60843373, 0.55394191, 1.02692308, 0.78070175, 1.12184874, 2.34210526],
        dtype=torch.float,
    )

    loss = WKLoss(num_classes, penalization_type, weight)

    input_data = torch.tensor(
        [
            [-2.4079, -2.5133, -1.6020, -2.2353, -3.2371, -4.6382],
            [-2.4079, -2.5133, -2.6187, -2.6082, -1.8703, -2.4251],
            [-2.4079, -1.8561, -3.9906, -5.2992, -5.8944, -7.1456],
        ]
    )

    target = torch.tensor([2, 4, 1])

    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() > 0


def test_wkloss_zeroloss():
    num_classes = 6
    weight = torch.tensor(
        [1.60843373, 0.55394191, 1.02692308, 0.78070175, 1.12184874, 2.34210526],
        dtype=torch.float,
    )

    loss_quadratic = WKLoss(num_classes, "quadratic", weight)
    loss_linear = WKLoss(num_classes, "linear", weight)

    input_data = torch.tensor(
        [
            [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10000.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10000.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 10000.0],
        ]
    )

    target = torch.tensor([0, 1, 2, 5])

    output_quadratic = loss_quadratic(input_data, target)
    output_linear = loss_linear(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output_quadratic, torch.Tensor)
    assert isinstance(output_linear, torch.Tensor)

    # Check that the loss is zero
    assert output_quadratic.item() == pytest.approx(0.0, rel=1e-6)
    assert output_linear.item() == pytest.approx(0.0, rel=1e-6)


def test_wkloss_exactloss_quadratic():
    num_classes = 6
    penalization_type = "quadratic"
    weight = torch.tensor(
        [1.60843373, 0.55394191, 1.02692308, 0.78070175, 1.12184874, 2.34210526],
        dtype=torch.float,
    )

    loss = WKLoss(num_classes, penalization_type, weight)

    input_data = torch.tensor(
        [
            [1.1182, -1.4991, -1.6639, -2.6338, 1.0638, -1.7242],
            [0.0584, 0.8491, 0.4281, -0.1311, 0.3988, 0.9868],
            [0.1456, -0.6615, -0.7971, -0.6217, 1.5191, 1.2822],
            [-1.1213, -0.1885, 0.4632, -1.0184, -0.4659, -0.0324],
            [-1.9973, -1.9622, -0.7560, -0.1352, 0.3598, 0.9245],
            [-0.4714, 0.9953, 0.3056, 1.4104, -0.3741, -1.4388],
            [-0.0293, -1.6450, -1.4326, -0.3240, 1.1304, 2.6315],
            [1.2635, 0.4663, -0.2953, 1.1983, -1.0992, -1.1091],
            [2.0113, 0.1522, 0.8675, -0.5035, -0.6438, 0.1304],
            [-0.4665, 1.9245, 1.6211, 0.7804, -0.9119, 0.8178],
        ]
    )

    target = torch.tensor([0, 4, 1, 3, 5, 4, 4, 3, 1, 2])

    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() == pytest.approx(1.1249, rel=1e-3)


def test_wkloss_exactloss_linear():
    num_classes = 6
    penalization_type = "linear"
    weight = torch.tensor(
        [1.60843373, 0.55394191, 1.02692308, 0.78070175, 1.12184874, 2.34210526],
        dtype=torch.float,
    )

    loss = WKLoss(num_classes, penalization_type, weight)

    input_data = torch.tensor(
        [
            [1.1182, -1.4991, -1.6639, -2.6338, 1.0638, -1.7242],
            [0.0584, 0.8491, 0.4281, -0.1311, 0.3988, 0.9868],
            [0.1456, -0.6615, -0.7971, -0.6217, 1.5191, 1.2822],
            [-1.1213, -0.1885, 0.4632, -1.0184, -0.4659, -0.0324],
            [-1.9973, -1.9622, -0.7560, -0.1352, 0.3598, 0.9245],
            [-0.4714, 0.9953, 0.3056, 1.4104, -0.3741, -1.4388],
            [-0.0293, -1.6450, -1.4326, -0.3240, 1.1304, 2.6315],
            [1.2635, 0.4663, -0.2953, 1.1983, -1.0992, -1.1091],
            [2.0113, 0.1522, 0.8675, -0.5035, -0.6438, 0.1304],
            [-0.4665, 1.9245, 1.6211, 0.7804, -0.9119, 0.8178],
        ]
    )

    target = torch.tensor([0, 4, 1, 3, 5, 4, 4, 3, 1, 2])

    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Verifies that the loss is greater than zero
    assert output.item() == pytest.approx(0.7434, rel=1e-3)


def test_wkloss_relative():
    num_classes = 6
    penalization_type = "quadratic"
    weight = torch.tensor(
        [1.60843373, 0.55394191, 1.02692308, 0.78070175, 1.12184874, 2.34210526],
        dtype=torch.float,
    )

    loss = WKLoss(num_classes, penalization_type, weight)

    # Predicting the correct category
    input_data1 = torch.tensor(
        [
            [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10000.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10000.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 10000.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 10000.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 10000.0],
        ]
    )

    # Predicting adjacent category
    input_data2 = torch.tensor(
        [
            [0.0, 10000.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10000.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 10000.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 10000.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 10000.0],
            [0.0, 0.0, 0.0, 0.0, 10000.0, 0.0],
        ]
    )

    # Predicting 2 categories away from target
    input_data3 = torch.tensor(
        [
            [0.0, 0.0, 10000.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 10000.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 10000.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 10000.0],
            [0.0, 0.0, 10000.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 10000.0, 0.0, 0.0],
        ]
    )

    target = torch.tensor([0, 1, 2, 3, 4, 5])

    output1 = loss(input_data1, target)
    output2 = loss(input_data2, target)
    output3 = loss(input_data3, target)

    assert isinstance(output1, torch.Tensor)
    assert isinstance(output2, torch.Tensor)
    assert isinstance(output3, torch.Tensor)

    assert output1.item() < output2.item() < output3.item()


def test_wkloss_penalization_types():
    num_classes = 6
    weight = torch.tensor(
        [1.60843373, 0.55394191, 1.02692308, 0.78070175, 1.12184874, 2.34210526],
        dtype=torch.float,
    )

    loss_quadratic = WKLoss(num_classes, "quadratic", weight)
    loss_linear = WKLoss(num_classes, "linear", weight)

    input_data = torch.tensor(
        [
            [1.1182, -1.4991, -1.6639, -2.6338, 1.0638, -1.7242],
            [0.0584, 0.8491, 0.4281, -0.1311, 0.3988, 0.9868],
            [0.1456, -0.6615, -0.7971, -0.6217, 1.5191, 1.2822],
            [-1.1213, -0.1885, 0.4632, -1.0184, -0.4659, -0.0324],
            [-1.9973, -1.9622, -0.7560, -0.1352, 0.3598, 0.9245],
            [-0.4714, 0.9953, 0.3056, 1.4104, -0.3741, -1.4388],
            [-0.0293, -1.6450, -1.4326, -0.3240, 1.1304, 2.6315],
            [1.2635, 0.4663, -0.2953, 1.1983, -1.0992, -1.1091],
            [2.0113, 0.1522, 0.8675, -0.5035, -0.6438, 0.1304],
            [-0.4665, 1.9245, 1.6211, 0.7804, -0.9119, 0.8178],
        ]
    )

    target = torch.tensor([5, 0, 0, 5, 1, 0, 2, 0, 5, 5])

    output_quadratic = loss_quadratic(input_data, target)
    output_linear = loss_linear(input_data, target)

    assert isinstance(output_quadratic, torch.Tensor)
    assert isinstance(output_linear, torch.Tensor)

    assert output_quadratic.item() > output_linear.item()


def test_wkloss_weights():
    num_classes = 6
    penalization_types = ["linear", "quadratic"]
    weight = torch.tensor(
        [1.0, 1.0, 1.0, 1.0, 1.0, 5.0],
        dtype=torch.float,
    )
    weight2 = torch.tensor(
        [1.0, 1.0, 1.0, 1.0, 1.0, 10.0],
        dtype=torch.float,
    )

    for penalization_type in penalization_types:
        loss = WKLoss(num_classes, penalization_type, weight=None)
        loss_weighted = WKLoss(num_classes, penalization_type, weight=weight)
        loss_weighted2 = WKLoss(num_classes, penalization_type, weight=weight2)

        # Input data with error in two samples of class 5 (highest weight)
        input_data = torch.tensor(
            [
                [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 10000.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 10000.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 10000.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 10000.0, 0.0],
                [0.0, 10000.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 10000.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        input_data2 = torch.tensor(
            [
                [10000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 10000.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 10000.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 10000.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 10000.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 10000.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 10000.0],
            ]
        )

        target = torch.tensor([0, 1, 2, 3, 4, 5, 5])

        output = loss(input_data, target)
        output_weighted = loss_weighted(input_data, target)
        output_weighted2 = loss_weighted2(input_data, target)

        output2 = loss(input_data2, target)
        output2_weighted = loss_weighted(input_data2, target)
        output2_weighted2 = loss_weighted2(input_data2, target)

        # Test return type
        assert isinstance(output, torch.Tensor)
        assert isinstance(output_weighted, torch.Tensor)
        assert isinstance(output_weighted2, torch.Tensor)

        # Test that using class weight with errors increases the loss
        assert output_weighted.item() > output.item()

        # When using a higher weight in the class with errors, the loss increases
        assert output_weighted2.item() > output_weighted.item()

        # When having an error in a class with not highest weight, the loss decreases
        assert output2_weighted.item() < output2.item()

        # If the cost of the other class increases, the loss decreases
        assert output2_weighted2.item() < output2_weighted.item()
