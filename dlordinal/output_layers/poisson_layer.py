from typing import Optional

import torch
from torch.distributions import Poisson


class PoissonLayer(torch.nn.Module):
    """
    Unimodal output layer for ordinal classification based on the Poisson distribution.
    Proposed by :footcite:t:`beckham2017unimodal`.

    Learns the λ parameter of the Poisson distribution from the input features and uses the
    Poisson distribution to compute the probabilities of each class, ensuring that the output
    is unimodal and that the probabilities sum to 1. The softplus of the linear layer output is
    used to ensure that the λ parameter is positive. Additionally, its value is clamped between
    1e-8 and 1e4 to prevent numerical issues.

    The layer includes an optional learnable temperature parameter τ that controls the sharpness
    of the output distribution. Higher values of τ produce softer distributions, while lower
    values produce sharper distributions. If learn_tau is set to False, τ is fixed at 1
    (no scaling).

    Parameters
    ----------
    in_features : int
        Size of the input feature vector (output features from the previous layer).

    num_classes : int
        Number of discrete output classes. Defines support of the distribution
        as {0, ..., num_classes - 1}.

    learn_tau : bool, default=True
        If True, the temperature parameter τ is learned as a model parameter.
        Otherwise, it is stored as a fixed buffer.

    Attributes
    ----------
    lambda_layer : torch.nn.Linear
        Linear transformation that maps input features to a scalar rate λ.

    log_tau : torch.Tensor or torch.nn.Parameter
        Log-temperature parameter used to control sharpness of the distribution.

    num_classes : int
        Number of output classes.

    learn_tau : bool
        Whether temperature is learnable.

    Example
    -------
    >>> import torch
    >>> from dlordinal.output_layers import PoissonLayer
    >>> layer = PoissonLayer(in_features=5, num_classes=3, learn_tau=True)
    >>> input = torch.randn(2, 5)
    >>> probs = layer(input)
    >>> print(probs)
    """

    log_tau: Optional[torch.Tensor]

    def __init__(self, *, in_features: int, num_classes: int, learn_tau: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.learn_tau = learn_tau
        tau_init = torch.tensor(1.0).log()
        if learn_tau:
            self.register_parameter("log_tau", torch.nn.Parameter(tau_init))
        else:
            self.register_buffer("log_tau", tau_init)
        self.lambda_layer = torch.nn.Linear(in_features, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute class probabilities using a Poisson-based discrete distribution.

        Parameters
        ----------
        input : torch.Tensor, shape (batch_size, in_features)
            Input feature tensor.

        Returns
        -------
        torch.Tensor, shape (batch_size, num_classes)
            Probability distribution over discrete classes.
        """

        # 1. Compute rate λ > 0
        lambda_ = torch.nn.functional.softplus(self.lambda_layer(input).squeeze(-1))
        lambda_ = lambda_.clamp(min=1e-8, max=1e4)

        # 2. Compute Poisson log-probabilities for all classes
        k = torch.arange(self.num_classes, device=input.device, dtype=input.dtype)
        pois = Poisson(rate=lambda_[:, None])
        log_probs = pois.log_prob(k)

        # 3. Temperature scaling (controls sharpness)
        tau = torch.exp(self.log_tau)
        scaled_log_probs = log_probs / tau

        # 4. Softmax normalisation
        probs = torch.softmax(scaled_log_probs, dim=1)
        return probs
