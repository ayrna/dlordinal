import torch


def stable_sigmoid(t: torch.Tensor) -> torch.Tensor:
    """
    Stable sigmoid function that avoids overflow issues for large values of t.
    This function is used to compute the sigmoid of a tensor t, handling both positive
    and negative values in a numerically stable way.

    Parameters
    ----------
    t : torch.Tensor
        Input tensor
    """
    # torch.sigmoid is already numerically stable for large |t|. It avoids the
    # boolean indexing of a manual implementation, which needs several kernels and
    # a host-device synchronisation on GPU.
    return torch.sigmoid(t)
