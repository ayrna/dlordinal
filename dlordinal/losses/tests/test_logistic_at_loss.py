import torch
import pytest
from dlordinal.losses import LogisticATLoss


def test_logistic_at_loss_creation():
    loss = LogisticATLoss(num_classes=5)
    assert isinstance(loss, LogisticATLoss)


def test_logistic_at_loss_basic():
    loss = LogisticATLoss(num_classes=4)
    loss_reg = LogisticATLoss(num_classes=4, reg_lambda=0.5)

    x = torch.tensor([0.5, -2, -1, 3, 4])
    w = torch.tensor([0.675])
    thresholds = torch.tensor([-0.7, 1.1, 2.3])
    target = torch.tensor([1, 2, 0, 2, 3])

    assert loss.forward((None, w, x, thresholds), target) == pytest.approx(
        7.05051912359419, rel=1e-6
    )
    assert loss_reg.forward((None, w, x, thresholds), target) == pytest.approx(
        7.16442537359419, rel=1e-6
    )
