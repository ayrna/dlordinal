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
            Targets of the batch. Must be a 1D tensor of shape (``batch_size``,)
            containing integer class indices for each sample in the batch.
            One-hot or soft label tensors of shape (``batch_size``, ``num_classes``) are
            not supported.

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

    The dropout is applied per unit (neuron): each unit of the input is either kept
    or dropped for the whole batch. The drop probability of unit ``i`` is

    .. math::
        p_i = \\beta \\left(1 - \\frac{1 + r_i}{2}\\right) + (1 - \\beta)\\, p

    where :math:`r_i` is the Pearson correlation between the activations of unit
    ``i`` and the batch targets. An undefined correlation (a unit or a target vector
    with zero variance in the batch) is treated as :math:`r_i = 0`, i.e. no ordinal
    preference. Kept units are scaled by :math:`1 / (1 - p_i)` (inverted dropout),
    so that the expected output of every unit during training matches its output in
    evaluation mode.

    To use this module, you must wrap your model with the ``HybridDropoutContainer``
    module. The module expects inputs of shape ``(N, features)``.

    Parameters
    ----------
    p : float
        Base probability of a unit (neuron) being dropped for the whole batch, used
        by the standard dropout component. Default: 0.5
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

    @staticmethod
    def _pearson_correlation(x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Pearson correlation between each column of ``x`` and ``targets``.

        Undefined correlations (zero variance) are returned as 0.

        Parameters
        ----------
        x : torch.Tensor
            Activations of shape ``(N, features)``.
        targets : torch.Tensor
            Targets of shape ``(N,)``.

        Returns
        -------
        torch.Tensor
            Correlations of shape ``(features,)`` on ``x.device``.
        """
        dtype = torch.promote_types(x.dtype, torch.float32)
        x = x.detach().to(dtype)
        t = targets.detach().to(device=x.device, dtype=dtype)

        xc = x - x.mean(dim=0, keepdim=True)
        tc = t - t.mean()
        numerator = tc @ xc
        denominator = torch.sqrt(xc.square().sum(dim=0) * tc.square().sum())
        r = numerator / denominator
        return torch.nan_to_num(r, nan=0.0).clamp(-1.0, 1.0)

    def _drop_probabilities(
        self, x: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Per-unit drop probabilities for the current batch.

        Parameters
        ----------
        x : torch.Tensor
            Activations of shape ``(N, features)``.
        targets : torch.Tensor
            Targets of shape ``(N,)``.

        Returns
        -------
        torch.Tensor
            Drop probability of each unit, shape ``(features,)``.
        """
        correlations = self._pearson_correlation(x, targets)

        # Rescale the correlation from [-1, 1] to [0, 1]
        correlations = (1 + correlations) / 2

        # Get ordinal probabilities
        ordinal_prob = 1 - correlations

        # The first summand is the one related to ordinal dropout and the second
        # summand is the standard dropout.
        return (self.beta * ordinal_prob) + ((1 - self.beta) * self.p)

    def forward(self, x):
        """Forward pass of the HybridDropout module just during training. The module
        calculates the correlation between the activation values of the neuron and the
        target labels of the dataset. Then, it calculates the ordinal probabilities
        and the mask for the dropout.

        Parameters
        ----------
            x : torch.Tensor
                Input tensor of shape ``(N, features)``.

        Raises
        ------
        ValueError
            If the input is not a 2D tensor.
        ValueError
            If there are NaN values in the tensor.
        ValueError
            If the batch targets have not been set or have the wrong shape.

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        if not self.training:
            return x

        if x.ndim != 2:
            raise ValueError(
                "HybridDropout expects an input of shape (N, features), but got a"
                f" {x.ndim}D tensor with shape {tuple(x.shape)}. Flatten the input"
                " before this layer."
            )

        if torch.isnan(x).any():
            raise ValueError("Nan values in the tensor")

        if hasattr(self, "batch_targets"):
            targets = self.batch_targets
        else:
            raise ValueError(
                "Batch targets have not been set. Use"
                " HybridDropoutContainer.set_targets() to set the targets."
            )

        if targets.ndim != 1:
            raise ValueError(
                "Targets must be a 1D tensor"
                f" but got {targets.ndim}D tensor. "
                "If you are using one-hot encoding or soft labels, (shape [batch_size, num_classes]),"
                " please convert them to class indices (shape [batch_size])"
            )

        if targets.shape[0] != x.shape[0]:
            raise ValueError(
                f"Targets have {targets.shape[0]} elements but the batch has"
                f" {x.shape[0]} samples. Call set_targets() for every batch."
            )

        with torch.no_grad():
            probabilities = self._drop_probabilities(x, targets)
            keep_prob = 1 - probabilities

            # One mask value per unit, shared by the whole batch
            mask = torch.rand(x.shape[1], device=x.device) >= probabilities

            # Inverted dropout with per-unit scaling: E[scale_i] = 1 for every unit.
            # Clamping avoids inf * 0 when a unit is always dropped.
            scale = mask.to(keep_prob.dtype) / keep_prob.clamp_min(1e-6)

        return x * scale.to(x.dtype)
