from typing import Optional

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F


class NormalizedCrossEntropyLoss(CrossEntropyLoss):
    r"""Cross entropy with a mean weighted by effective target mass.

    The mean reduction divides the summed cross entropy by
    :math:`\sum_{n,c} q_{n,c} w_c`, where :math:`q` is the target distribution
    after label smoothing. Class indices represent one-hot distributions. Spatial
    positions are included in the sum, and ignored class indices contribute zero
    target mass. Each class has weight one when ``weight=None``.

    This denominator makes the mean invariant to positive rescaling of class
    weights. Hard and one-hot targets produce the same result, including when
    label smoothing is enabled. For hard targets, PyTorch's
    :class:`torch.nn.CrossEntropyLoss` uses the original target class weights in
    its mean denominator when smoothing is enabled.

    Parameters follow :class:`torch.nn.CrossEntropyLoss`. ``weight`` must be a
    nonempty, finite, nonnegative vector. Probability targets must satisfy
    PyTorch's probability-target requirements. ``ignore_index`` applies to class
    indices. The ``none`` and ``sum`` reductions follow PyTorch. A mean with zero
    effective target weight returns zero with zero input gradients.

    Mean reduction accumulates in float32 for float16 and bfloat16 inputs. The
    returned scalar has float32 precision for these inputs. Input gradients retain
    the input precision.

    Example
    -------
    >>> from dlordinal.losses import BetaLoss, NormalizedCrossEntropyLoss
    >>> base_loss = NormalizedCrossEntropyLoss(weight=torch.tensor([1., 2., 3.]))
    >>> loss = BetaLoss(base_loss, num_classes=3)
    """

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        if weight is not None:
            if weight.ndim != 1 or weight.numel() == 0:
                raise ValueError("weight must be a nonempty vector of class weights")
            if not torch.isfinite(weight).all() or (weight < 0).any():
                raise ValueError("weight must contain finite, nonnegative values")
        super().__init__(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def _normalized_weight(
        self, input: Tensor, target: Tensor, weight: Tensor
    ) -> Tensor:
        class_dim = 0 if input.ndim == 1 else 1
        num_classes = input.shape[class_dim]
        if target.shape == input.shape:
            if target.requires_grad:
                active = torch.ones_like(weight, dtype=torch.bool)
            else:
                mass = (1.0 - self.label_smoothing) * target
                mass = mass + self.label_smoothing / num_classes
                active = mass != 0
                if input.ndim > 1:
                    sample_dims = tuple(i for i in range(input.ndim) if i != class_dim)
                    active = active.any(dim=sample_dims)
        else:
            indices = target.reshape(-1).long()
            valid = indices != self.ignore_index
            if self.label_smoothing > 0:
                active = torch.ones_like(weight, dtype=torch.bool) & valid.any()
            else:
                indices = indices[valid].clamp(0, num_classes - 1)
                active = torch.bincount(indices, minlength=num_classes) > 0

        effective_weight = weight.masked_fill(~active, 0)
        scale = effective_weight.max()
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return effective_weight / scale

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        weight = self.weight
        if self.reduction == "mean" and input.dtype in (torch.float16, torch.bfloat16):
            input = input.float()
            if weight is not None:
                weight = weight.float()
            if target.is_floating_point():
                target = target.float()

        if self.reduction == "mean" and weight is not None:
            weight = self._normalized_weight(input, target, weight)

        reduction = "sum" if self.reduction == "mean" else self.reduction
        loss = F.cross_entropy(
            input,
            target,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction=reduction,
            label_smoothing=self.label_smoothing,
        )
        if self.reduction != "mean":
            return loss

        class_dim = 0 if input.ndim == 1 else 1
        num_classes = input.shape[class_dim]
        if target.shape == input.shape:
            target_mass = (1.0 - self.label_smoothing) * target.to(loss.dtype)
            target_mass = target_mass + self.label_smoothing / num_classes
            if weight is not None:
                weight_shape = [1] * input.ndim
                weight_shape[class_dim] = num_classes
                target_mass = target_mass * weight.view(weight_shape)
            total_weight = target_mass.sum()
        else:
            target_indices = target.long()
            valid = target_indices != self.ignore_index
            if weight is None:
                total_weight = valid.to(input.dtype).sum()
            else:
                safe_target = target_indices.masked_fill(~valid, 0)
                target_mass = (1.0 - self.label_smoothing) * weight[safe_target]
                target_mass = target_mass + self.label_smoothing * weight.mean()
                total_weight = target_mass.masked_fill(~valid, 0).sum()

        positive = total_weight > 0
        denominator = torch.where(positive, total_weight, torch.ones_like(total_weight))
        return torch.where(positive, loss / denominator, loss * 0.0)
