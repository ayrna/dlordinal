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
    idx = t > 0
    out = torch.zeros_like(t)
    out[idx] = 1.0 / (1 + torch.exp(-t[idx]))
    exp_t = torch.exp(t[~idx])
    out[~idx] = exp_t / (1.0 + exp_t)
    return out
