import torch
import torch.nn as nn


class HybridDropoutContainer(nn.Module):
    """Container for the ``HybridDropout`` module. This container is used to set the
    targets of the batch in the HybridDropout module.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be wrapped.
    """

    def __init__(self, model):
        super(HybridDropoutContainer, self).__init__()
        self.model = model

    def forward(self, x):
        """Forward pass of the ``HybridDropoutContainer`` module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """

        return self.model(x)

    def set_targets(self, targets):
        """
        Set the targets of the batch in the ``HybridDropout`` module.

        Parameters
        ----------
        targets : torch.Tensor
            Targets of the batch

        Example
        -------
        >>> from dlordinal.dropout import HybridDropoutContainer
        >>> from torchvision.models import resnet18
        >>> model = resnet18(weights='IMAGENET1K_V1')
        >>> model.fc = nn.Sequential(
                nn.Linear(model.fc.in_features, 256),
                HybridDropout(),
                nn.Linear(256, num_classes),
            )
        >>> batch_targets = torch.tensor([...])
        >>> model = HybridDropoutContainer(model)
        >>> model.set_targets(targets)
        """

        for module in self.model.modules():
            if isinstance(module, HybridDropout):
                module.batch_targets = targets


class HybridDropout(nn.Module):
    """Implements a hybrid dropout methodology by :footcite:t:`berchez2024fusion` which
    mix a standard dropout with an ordinal dropout. The ordinal dropout is based on the
    correlation between the activation values of the neuron and the target labels
    of the dataset.

    To use this module, you must wrap your model with the ``HybridDropoutContainer``
    module

    Parameters
    ----------
    p : float
        Probability of an element to be zeroed. Default: 0.5
    beta : float
        Weight of the ordinal dropout. Default: 0.1
    batch_targets : torch.Tensor
        Targets of the batch. Default: None

    Raises
    ------
    ValueError
        If ``p`` is not a probability.
    """

    def __init__(self, p: float = 0.5, beta: float = 0.1):
        super(HybridDropout, self).__init__()
        self.p = p
        self.beta = beta
        if self.p < 0 or self.p > 1:
            raise ValueError("p must be a probability")

    def forward(self, x):
        """Forward pass of the HybridDropout module just during training. The module
        calculates the correlation between the activation values of the neuron and the
        target labels of the dataset. Then, it calculates the ordinal probabilities
        and the mask for the dropout.

        Parameters
        ----------
            x : torch.Tensor
                Input tensor

        Raises
        ------
        ValueError
            If there are NaN values in the tensor.
        ValueError
            If the batch targets have not been set.

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        if self.training:
            if torch.isnan(x).any():
                raise ValueError("Nan values in the tensor")

            if hasattr(self, "batch_targets"):
                targets = self.batch_targets
            else:
                raise ValueError(
                    "Batch targets have not been set. Use"
                    " HybridDropoutContainer.set_targets() to set the targets."
                )

            targets = torch.reshape(targets, (1, targets.shape[0]))

            correlation_list = []

            # Pearson's correlation calculation for each neuron
            for neuron in range(0, x.shape[1]):
                patterns = x[:, neuron]
                patterns = torch.reshape(patterns, (1, x.shape[0]))
                concat = torch.cat((patterns, targets), 0)
                corr = torch.corrcoef(concat)
                correlation_list.append(float(corr[0, 1]))

            correlations = torch.Tensor(correlation_list)

            # Scale of the correlation matrix
            correlations = 1 + correlations
            correlations = correlations / 2
            correlations = torch.nan_to_num(correlations)

            # correlations = correlations.to(x.device)

            # Get ordinal probabilities
            ordinal_prob = 1 - correlations

            # Mask creation: the first summand is the one related to ordinal dropout and
            # the second summand is the standard dropout.
            probabilities = (self.beta * ordinal_prob) + ((1 - self.beta) * self.p)
            mask = torch.empty(x.size()[1]).uniform_(0, 1) >= probabilities
            mask = mask.to(x.device)

            # Normalisation
            no_zeros = int(torch.count_nonzero(mask))
            total_neurons = mask.shape[0]
            zeros = total_neurons - no_zeros
            probability = zeros / total_neurons

            mask = torch.reshape(mask, (1, mask.shape[0]))
            mask = mask.repeat(x.shape[0], 1)

            return x.mul(mask) * probability
        else:
            return x
