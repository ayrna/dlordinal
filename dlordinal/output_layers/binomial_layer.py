import torch
from torch.distributions import Binomial


class BinomialLayer(torch.nn.Module):
    """
    Unimodal output layer for ordinal classification based on the binomial distribution.
    Proposed by :footcite:t:`beckham2017unimodal`.

    Learns the p parameter of the binomial distribution from the input features and uses the
    binomial distribution to compute the probabilities of each class, ensuring that the output
    is unimodal and that the probabilities sum to 1. The sigmoid of the linear layer output is
    used to ensure that the p parameter is between 0 and 1.

    Parameters
    ----------
    in_features : int
        Number of input features (output features from the previous layer).

    num_classes : int
        Number of output classes. Defines the support of the binomial
        distribution (0 to num_classes - 1).

    Attributes
    ----------
    p_layer : torch.nn.Linear
        Linear layer that maps input features to a scalar logit.

    num_classes : int
        Number of classes used to define the binomial distribution.

    Example
    -------
    >>> import torch
    >>> from dlordinal.output_layers import BinomialLayer
    >>> layer = BinomialLayer(in_features=5, num_classes=3)
    >>> input = torch.randn(2, 5)
    >>> probs = layer(input)
    >>> print(probs)
    """

    def __init__(self, *, in_features: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.p_layer = torch.nn.Linear(in_features, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute class probabilities using a binomial distribution.

        Parameters
        ----------
        input : torch.Tensor, shape (batch_size, in_features)
            Input feature tensor.

        Returns
        -------
        torch.Tensor, shape (batch_size, num_classes)
            Probability distribution over classes.
        """

        num_classes = self.num_classes

        p = torch.sigmoid(self.p_layer(input)).squeeze(-1)
        k = torch.arange(num_classes, device=input.device, dtype=input.dtype)

        binom = Binomial(total_count=num_classes - 1, probs=p[:, None])
        probs = torch.exp(binom.log_prob(k))

        return probs
