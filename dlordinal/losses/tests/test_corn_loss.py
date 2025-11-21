import pytest
import torch
from torch import nn

from dlordinal.losses import CORNLoss

# Use a high precision for checks
ATOL = 1e-6


@pytest.fixture
def corn_loss_fn():
    """Fixture to initialize CORNLoss with 5 classes (0 to 4)."""
    return CORNLoss(num_classes=5)


@pytest.fixture
def num_classes():
    return 5


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ]
)
def device(request):
    return torch.device(request.param)


def test_cornloss_creation(device):
    """
    Tests successful initialization of CORNLoss, ensuring the correct number
    of classes is stored and the module is moved to the specified device.
    """
    NUM_CLASSES = 6
    loss = CORNLoss(num_classes=NUM_CLASSES).to(device)

    assert isinstance(loss, CORNLoss)
    assert loss.num_classes == NUM_CLASSES
    # Verifies the module is on the correct device
    assert next(loss.parameters(), torch.tensor(0.0)).device.type == device.type


def test_cornloss_exact_value(device):
    """
    Tests a basic case, verifying the loss output against a known, pre-calculated
    reference value to check for exact logical correctness (replacing loss > 0 check).
    """
    num_classes = 6
    ATOL = 1e-6  # Tolerance for floating point comparison

    loss_fn = CORNLoss(num_classes).to(device)

    # Fixed input data and targets (Batch size N=3, J-1=5 logits)
    y_pred = torch.tensor(
        [
            [-2.4079, -2.5133, -2.6187, -2.0652, -3.7299],
            [-2.4079, -2.1725, -2.1459, -3.3318, -3.9624],
            [-2.4079, -1.7924, -2.0101, -4.1030, -3.3445],
        ],
        dtype=torch.float32,
    ).to(device)

    y_true = torch.tensor([2, 2, 1]).to(device)

    # 1. Calculate the expected loss using the reference function (on CPU for safety)
    expected_loss = calculate_ref_corn_loss(y_pred.cpu(), y_true.cpu(), num_classes).to(
        device
    )

    # 2. Calculate the actual loss using the CORNLoss class
    actual_loss = loss_fn(y_pred, y_true)

    # Assertion: Check that the result is identical to the reference calculation
    assert torch.isclose(actual_loss, expected_loss, atol=ATOL)
    assert (
        actual_loss.item() > 0
    )  # Also ensures the loss is not zero (unless it should be)


def test_cornloss_zeroloss(device):
    """
    Tests the ideal edge case where the loss should be zero (perfect prediction).
    Target: [0, 1, 2] for 3 classes (J=3, J-1=2 tasks).
    Logits:
    - y=0: [very negative, very negative] -> Targets [0, 0] -> Correct
    - y=1: [very positive, very negative] -> Targets [1, 0] -> Correct
    - y=2: [very positive, very positive] -> Targets [1, 1] -> Correct
    """
    num_classes = 3

    loss = CORNLoss(num_classes).to(device)

    zero = -1e3  # Logit that maps to P(y>=k) ~ 0
    one = 1e3  # Logit that maps to P(y>=k) ~ 1

    input_data = torch.tensor(
        [
            [zero, zero],  # Target 0 -> Binary Targets [0, 0]
            [one, zero],  # Target 1 -> Binary Targets [1, 0]
            [one, one],  # Target 2 -> Binary Targets [1, 1]
        ]
    ).to(device)

    target = torch.tensor([0, 1, 2]).to(device)

    output = loss(input_data, target)

    # Verifies that the output is a tensor
    assert isinstance(output, torch.Tensor)

    # Check that the loss is zero (within tolerance)
    assert output.item() == pytest.approx(0.0, rel=1e-6)


# --- Helper Function for External Validation ---
def calculate_ref_corn_loss(y_pred, y_true, num_classes):
    """Calculates the reference loss using nn.BCEWithLogitsLoss for comparison."""
    total_loss = 0.0
    total_elements = 0

    # Iterate over J-1 tasks
    for i in range(num_classes - 1):
        # 1. Masking: Only include samples where y_true >= i (y_true > i - 1)
        mask = y_true > i - 1

        # 2. Binary Target: 1 if y_true > i, 0 if y_true = i
        target = (y_true[mask] > i).float()

        # 3. Prediction: Logits for the i-th task
        pred_i = y_pred[mask, i]

        if len(target) > 0:
            loss_i = nn.functional.binary_cross_entropy_with_logits(
                pred_i, target, reduction="sum"
            )
            total_loss += loss_i
            total_elements += len(target)

    return total_loss / total_elements


def test_corn_loss_vs_reference(corn_loss_fn, num_classes):
    """Tests that the loss matches the manual calculation of BCE with CORN masking."""

    # Batch size N=4. Classes J=5 (0, 1, 2, 3, 4). Logits shape (4, 4).
    y_pred = torch.tensor(
        [
            [1.0, 0.5, 0.0, -1.0],  # Sample 0
            [-1.0, -0.5, 0.5, 1.0],  # Sample 1
            [0.0, 0.0, 0.0, 0.0],  # Sample 2
            [2.0, -2.0, 2.0, -2.0],  # Sample 3
        ],
        dtype=torch.float32,
    )
    # True labels: [Class 0, Class 4, Class 1, Class 2]
    y_true = torch.tensor([0, 4, 1, 2], dtype=torch.int64)

    # Calculate the reference loss
    expected_loss = calculate_ref_corn_loss(y_pred, y_true, num_classes)

    # Calculate the loss using the CORNLoss class
    actual_loss = corn_loss_fn(y_pred, y_true)

    assert torch.isclose(actual_loss, expected_loss, atol=ATOL)


def test_corn_loss_min_class(corn_loss_fn, num_classes):
    """Tests the edge case where all labels are the minimum class (0)."""

    # Logits for 3 samples. J-1 = 4 tasks.
    y_pred = torch.randn(3, num_classes - 1)
    y_true = torch.tensor([0, 0, 0], dtype=torch.int64)

    # Expected Logic: Only Task 0 (i=0, predicting y > 0) is active (mask: y_true >= 0).
    # The target for Task 0 is 0 for all. Total examples = 3.

    # Reference loss: Only task 0 (i=0) is considered with target=0
    expected_loss = (
        nn.functional.binary_cross_entropy_with_logits(
            y_pred[:, 0], torch.zeros(3).float(), reduction="sum"
        )
        / 3  # Divided by 3 total examples
    )

    actual_loss = corn_loss_fn(y_pred, y_true)

    assert torch.isclose(actual_loss, expected_loss, atol=ATOL)


def test_corn_loss_max_class(corn_loss_fn, num_classes):
    """Tests the edge case where all labels are the maximum class (J-1)."""

    y_pred = torch.randn(2, num_classes - 1)
    max_label = num_classes - 1  # max_label is 4 for num_classes=5
    y_true = torch.tensor([max_label, max_label], dtype=torch.int64)

    # Expected Logic: All samples participate in all J-1 tasks.
    # The target for all tasks is 1. Total examples = 8.

    # Reference loss: Sum of BCE over all 4 tasks with target=1
    total_loss_sum = 0.0
    for i in range(num_classes - 1):
        total_loss_sum += nn.functional.binary_cross_entropy_with_logits(
            y_pred[:, i], torch.ones(2).float(), reduction="sum"
        )

    expected_loss = total_loss_sum / 8  # Divided by 8 total examples

    actual_loss = corn_loss_fn(y_pred, y_true)

    assert torch.isclose(actual_loss, expected_loss, atol=ATOL)


def test_corn_loss_specific_mapping(corn_loss_fn, num_classes):
    """Tests a case where the label is exactly equal to the task index i for some tasks."""

    y_pred = torch.randn(1, num_classes - 1)  # One sample
    y_true = torch.tensor([2], dtype=torch.int64)  # True label is 2

    # Logic for y_true = 2 (J=5, tasks i=0, 1, 2, 3):
    # i=0 (y>0): target=1. Pred: y_pred[0, 0].
    # i=1 (y>1): target=1. Pred: y_pred[0, 1].
    # i=2 (y>2): target=0. Pred: y_pred[0, 2].
    # i=3 (y>3): mask=False. No loss.
    # Total examples = 3.

    # Reference Loss Calculation for the 3 active tasks
    target_i0 = torch.tensor([1.0])
    target_i1 = torch.tensor([1.0])
    target_i2 = torch.tensor([0.0])

    # Needs to be unsqueezed to match the target dimension for BCE function
    loss_i0 = nn.functional.binary_cross_entropy_with_logits(
        y_pred[0, 0].unsqueeze(0), target_i0, reduction="sum"
    )
    loss_i1 = nn.functional.binary_cross_entropy_with_logits(
        y_pred[0, 1].unsqueeze(0), target_i1, reduction="sum"
    )
    loss_i2 = nn.functional.binary_cross_entropy_with_logits(
        y_pred[0, 2].unsqueeze(0), target_i2, reduction="sum"
    )

    # Loss is normalized by the total number of sub-examples (3)
    expected_loss = (loss_i0 + loss_i1 + loss_i2) / 3

    actual_loss = corn_loss_fn(y_pred, y_true)

    # Final assertion
    assert torch.isclose(actual_loss, expected_loss, atol=ATOL)
