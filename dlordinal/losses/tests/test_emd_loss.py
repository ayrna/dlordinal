import pytest
import torch

from dlordinal.losses import EMDLoss
from dlordinal.metrics import ranked_probability_score


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_emd_loss_creation(device):
    loss = EMDLoss(num_classes=5).to(device)
    assert isinstance(loss, EMDLoss)


def test_emd_max(device):
    loss = EMDLoss(num_classes=3).to(device)

    y_pred = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).to(device)

    # Calculate logits from softmax probabilities
    y_pred_logits = torch.log(y_pred)

    y_true = torch.tensor([2, 0]).to(device)

    expected_rps = 2.0

    assert loss.forward(y_pred_logits, y_true).cpu().item() == pytest.approx(
        expected_rps, rel=1e-6
    )


def test_emd_min(device):
    loss = EMDLoss(num_classes=3).to(device)

    y_pred = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).to(device)

    # Calculate logits from softmax probabilities
    y_pred_logits = torch.log(y_pred)

    y_true = torch.tensor([0, 2]).to(device)

    expected_rps = 0.0

    assert loss.forward(y_pred_logits, y_true).cpu().item() == pytest.approx(
        expected_rps, rel=1e-6
    )


def test_emd_rps_loss(device):
    loss = EMDLoss(num_classes=4).to(device)

    y_true = torch.tensor([0, 0, 3, 2]).to(device)
    y_pred = torch.tensor(
        [
            [0.2, 0.4, 0.2, 0.2],
            [0.7, 0.1, 0.1, 0.1],
            [0.5, 0.05, 0.1, 0.35],
            [0.1, 0.05, 0.65, 0.2],
        ]
    ).to(device)

    # Calculate logits from softmax probabilities
    y_pred_logits = torch.log(y_pred)

    # Note: this test also shows that squared EMD loss is equivalent to RPS metric
    assert (
        loss.forward(y_pred_logits, y_true).cpu().item()
        == pytest.approx(0.506875, rel=1e-6)
        == ranked_probability_score(y_true.cpu(), y_pred.cpu())
    )
