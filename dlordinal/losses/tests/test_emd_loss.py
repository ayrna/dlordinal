import pytest
import torch

from dlordinal.losses import EMDLoss
from dlordinal.metrics import ranked_probability_score


def test_emd_loss_creation():
    loss = EMDLoss(num_classes=5)
    assert isinstance(loss, EMDLoss)


def test_emd_max():
    loss = EMDLoss(num_classes=3)

    y_pred = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # Calculate logits from softmax probabilities
    y_pred_logits = torch.log(y_pred)

    y_true = torch.tensor([2, 0])

    expected_rps = 2.0

    assert loss.forward(y_pred_logits, y_true).item() == pytest.approx(
        expected_rps, rel=1e-6
    )


def test_emd_min():
    loss = EMDLoss(num_classes=3)

    y_pred = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # Calculate logits from softmax probabilities
    y_pred_logits = torch.log(y_pred)

    y_true = torch.tensor([0, 2])

    expected_rps = 0.0

    assert loss.forward(y_pred_logits, y_true).item() == pytest.approx(
        expected_rps, rel=1e-6
    )


def test_emd_rps_loss():
    loss = EMDLoss(num_classes=4)

    y_true = torch.tensor([0, 0, 3, 2])
    y_pred = torch.tensor(
        [
            [0.2, 0.4, 0.2, 0.2],
            [0.7, 0.1, 0.1, 0.1],
            [0.5, 0.05, 0.1, 0.35],
            [0.1, 0.05, 0.65, 0.2],
        ]
    )

    # Calculate logits from softmax probabilities
    y_pred_logits = torch.log(y_pred)

    # Note: this test also shows that squared EMD loss is equivalent to RPS metric
    assert (
        loss.forward(y_pred_logits, y_true)
        == pytest.approx(0.506875, rel=1e-6)
        == ranked_probability_score(y_true, y_pred)
    )
