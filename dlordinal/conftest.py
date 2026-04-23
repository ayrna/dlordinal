import os

import pytest
import torch


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    cuda_only = os.getenv("PYTEST_CUDA_ONLY", "0") == "1"

    dev = request.param

    # CI GPU mode → skip CPU tests
    if cuda_only and dev != "cuda":
        pytest.skip("Skipping CPU tests in CUDA-only CI mode")

    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    return torch.device(dev)
