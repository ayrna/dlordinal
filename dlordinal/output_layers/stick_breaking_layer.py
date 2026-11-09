import torch
import torch.nn.functional as F
from torch.nn import Module


def _stick_breaking_log_probs(
    log_break: torch.Tensor, log_rest: torch.Tensor
) -> torch.Tensor:
    """Combine the log break fractions of a stick into class log-probabilities.

    Parameters
    ----------
    log_break : torch.Tensor
        :math:`\\log \\sigma_k`, tensor of shape ``(N, J - 1)``.
    log_rest : torch.Tensor
        :math:`\\log (1 - \\sigma_k)`, tensor of shape ``(N, J - 1)``.

    Returns
    -------
    log_probs : torch.Tensor
        Tensor of shape ``(N, J)`` with
        :math:`\\log p_k = \\log \\sigma_k + \\sum_{j<k} \\log (1 - \\sigma_j)` for
        :math:`k < J` and :math:`\\log p_J = \\sum_{j<J} \\log (1 - \\sigma_j)`.
    """
    # Log-length of the stick that is left before each break (exclusive cumsum).
    log_remaining = F.pad(torch.cumsum(log_rest, dim=1), (1, 0))
    # The last class keeps the whole remaining stick, i.e. a break fraction of 1.
    return F.pad(log_break, (0, 1)) + log_remaining


class StickBreakingLayer(Module):
    """Neuron stick-breaking output layer from :footcite:t:`liu2020unimodal`.

    A unit-length stick is broken ``J - 1`` times. Each break uses the sigmoid output
    :math:`\\sigma_k` of one neuron, so the class probabilities are

    .. math::
        p_1 = \\sigma_1, \\qquad
        p_k = \\sigma_k \\prod_{j=1}^{k-1} (1 - \\sigma_j), \\qquad
        p_J = \\prod_{j=1}^{J-1} (1 - \\sigma_j),

    which are non-negative and sum to one by construction. The layer returns
    :math:`\\log p_k`, computed in log space with
    :func:`torch.nn.functional.logsigmoid`, so it stays finite and keeps its
    gradients for saturated neurons. As the output is already a normalised
    log-probability, it can be passed as logits to
    :class:`torch.nn.CrossEntropyLoss` (``log_softmax`` leaves it unchanged) or to
    :class:`torch.nn.NLLLoss`.

    Parameters
    ----------
    input_shape: int
        Input shape, which refers to the number of neurons in the last fully connected
        layer.
    num_classes: int
        Number of classes :math:`J`. The layer has :math:`J - 1` output neurons.

    Raises
    ------
    ValueError
        If ``num_classes`` is lower than 2.

    Example
    -------
    >>> import torch
    >>> from dlordinal.output_layers import StickBreakingLayer
    >>> layer = StickBreakingLayer(input_shape=10, num_classes=5)
    >>> log_probs = layer(torch.randn(3, 10))
    >>> log_probs.shape
    torch.Size([3, 5])
    >>> torch.allclose(log_probs.exp().sum(dim=1), torch.ones(3))
    True
    """

    def __init__(self, input_shape: int, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, but got {num_classes}")
        self.num_classes = num_classes
        self.fcn1 = torch.nn.Linear(input_shape, num_classes - 1)

    def get_stick_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the class log-probabilities from the break fractions.

        Parameters
        ----------
        x : torch.Tensor
            Break fractions :math:`\\sigma_k \\in [0, 1]`, tensor of shape
            ``(N, num_classes - 1)``.

        Returns
        -------
        logits : torch.Tensor
            Class log-probabilities, tensor of shape ``(N, num_classes)``. A break
            fraction of exactly 0 or 1 gives ``-inf`` for the classes that cannot
            occur; :meth:`forward` avoids this by working from the pre-activations.
        """
        return _stick_breaking_log_probs(torch.log(x), torch.log1p(-x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(N, input_shape)``.

        Returns
        -------
        logits : torch.Tensor
            Class log-probabilities, tensor of shape ``(N, num_classes)``.
        """
        z = self.fcn1(x)
        return _stick_breaking_log_probs(F.logsigmoid(z), F.logsigmoid(-z))
