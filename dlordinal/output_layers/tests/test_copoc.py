import numpy as np
import pytest
import torch

from dlordinal.output_layers import COPOC


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _is_unimodal(probs):
    """Check if a 1D array is unimodal (increases to a peak, then decreases)."""
    peak_idx = np.argmax(probs)
    # Increasing up to peak
    inc = np.all(np.diff(probs[: peak_idx + 1]) >= 0)
    # Decreasing after peak
    dec = np.all(np.diff(probs[peak_idx:]) <= 0)
    return inc and dec


def _check_unimodality(y_pred):
    """Check unimodality for each row in y_pred and return the proportion."""
    unimodal_flags = np.array([_is_unimodal(row) for row in y_pred])
    # Proportion of rows that are unimodal
    proportion = np.mean(unimodal_flags)
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
    probs = layer(input_data)

    # Check that probabilities have the expected shape
    assert probs.shape == (num_rows, num_classes)

    # Check for unimodality
    assert _check_unimodality(probs) == 1.0
