import pytest
import torch

from dlordinal.output_layers import COPOC


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _is_unimodal(probs: torch.Tensor) -> bool:
    """
    Check if a 1D tensor is unimodal (increases to a peak, then decreases).
    """
    if probs.dim() != 1:
        raise ValueError("probs must be a 1D tensor")

    # Find the index of the peak
    peak_idx = torch.argmax(probs)

    if peak_idx.item() == 0:
        inc = True
    else:
        inc = torch.all(torch.diff(probs[: peak_idx + 1]) >= 0).item()

    if peak_idx.item() == (probs.numel() - 1):
        dec = True
    else:
        dec = torch.all(torch.diff(probs[peak_idx:]) <= 0).item()
    return bool(inc and dec)


def _check_unimodality(y_pred: torch.Tensor) -> float:
    """
    Check unimodality for each row in y_pred and return the proportion.

    y_pred: 2D tensor of shape (n_rows, n_cols)
    Returns: scalar float proportion of unimodal rows.
    """
    if y_pred.dim() != 2:
        raise ValueError("y_pred must be a 2D tensor")

    # Apply per-row unimodality check
    unimodal_flags = torch.tensor(
        [_is_unimodal(row) for row in y_pred], dtype=torch.bool, device=y_pred.device
    )

    proportion = unimodal_flags.float().mean().item()
    return proportion


def test_copoc_creation(device):
    copoc = COPOC().to(device)
    assert isinstance(copoc, COPOC)


def test_copoc_layer(device):
    num_classes = 4
    num_rows = 10

    layer = COPOC().to(device)

    input_data = torch.randn(num_rows, num_classes).to(device)

    # Compute unimodal probabilities from COPOC layer
    probs = torch.nn.functional.softmax(layer(input_data), dim=1)

    # Check that probabilities have the expected shape
    assert probs.shape == (num_rows, num_classes)

    # Check for unimodality
    assert _check_unimodality(probs) == 1.0
