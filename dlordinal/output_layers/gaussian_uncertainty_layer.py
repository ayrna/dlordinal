import torch
import torch.nn as nn


class GaussianUncertaintyLayer(nn.Module):
    """
    Discretized Gaussian Uncertainty (GU) layer proposed by :footcite:t:`araujo2020dr`.

    Produces a discrete unimodal distribution over `num_classes` classes,
    derived from a Gaussian with learnable mean (mu) and standard deviation (sigma).

    Parameters
    ----------
    in_features : int
        Number of input features (output of the previous layer).

    num_classes : int
        Number of discrete output classes.

    Attributes
    ----------
    num_classes : int
        Number of discrete output classes.

    mu_layer : nn.Linear
        Linear layer used to predict the mean (mu) of the Gaussian.

    sigma_layer : nn.Linear
        Linear layer used to predict the standard deviation (sigma).

    Notes
    -----
    - The standard deviation is parameterised in an unconstrained way and
      transformed using `softplus` to ensure positivity.
    - The resulting values are normalised to form a valid probability distribution.

    Example
    -------
    >>> layer = GaussianUncertaintyLayer(in_features=5, num_classes=3)
    >>> input = torch.randn(2, 5)
    >>> probs, sigma = layer(input)
    >>> print(probs)
    """

    mu: torch.Tensor
    log_sigma: torch.Tensor

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.mu_layer = nn.Linear(in_features, 1)
        self.sigma_layer = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, in_features)`.

        Returns
        -------
        probs : torch.Tensor
            Discrete probability distribution over classes.
            Shape: `(batch_size, num_classes)`.

        sigma : torch.Tensor
            Predicted standard deviation for each sample.
            Shape: `(batch_size,)`.
        """

        # 1. Predict mu and sigma
        mu = self.mu_layer(x).squeeze(-1)
        sigma = self.sigma_layer(x).squeeze(-1)
        sigma = torch.nn.functional.softplus(sigma).clamp(min=1e-3, max=1e3)

        # 2. Class indices
        k = torch.arange(self.num_classes, device=x.device, dtype=x.dtype)

        # 3. Compute probabilities using Gaussian PDF
        gaussian = torch.distributions.Normal(
            loc=mu[:, None], scale=sigma[:, None], validate_args=False
        )
        log_probs = gaussian.log_prob(k[None, :])

        # 4. Normalize to get probabilities. Normalising in log space (softmax) is
        # exact; exp(log_probs) underflows to all zeros for small sigma, which
        # gave zero probabilities and an infinite loss.
        probs = torch.softmax(log_probs, dim=-1)

        return probs, sigma
