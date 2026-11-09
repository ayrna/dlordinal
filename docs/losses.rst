.. _losses:

Loss functions
==============

Weighted means for imbalanced datasets
---------------------------------------------

``NormalizedCrossEntropyLoss`` divides the summed cross entropy by the total effective target weight. For target probabilities :math:`q_{ic}`, class weights :math:`w_c`, and predicted probabilities :math:`p_{ic}`, the mean is

.. math::

    L = \frac{-\sum_i\sum_c q_{ic}w_c\log p_{ic}}
             {\sum_i\sum_c q_{ic}w_c}.

The target distribution includes any label smoothing. For class indices, the distribution is the smoothed one hot representation of each target. Thus, class indices and equivalent probability targets produce the same loss. Multiplying every class weight by the same positive constant preserves the loss and its gradients. Batches can still differ in their loss because their targets and prediction errors differ.

Every soft labelling loss that accepts ``base_loss`` can use this criterion. The following example combines the normalized mean with Beta targets:

.. code-block:: python

    import torch
    from dlordinal.losses import BetaLoss, NormalizedCrossEntropyLoss

    weights = torch.tensor([1.0, 2.0, 4.0, 8.0])
    criterion = BetaLoss(
        base_loss=NormalizedCrossEntropyLoss(weight=weights),
        num_classes=4,
    )
    logits = torch.randn(3, 4, requires_grad=True)
    targets = torch.tensor([0, 1, 3])
    loss = criterion(logits, targets)

``CDWCELoss`` provides ``reduction="weighted_mean"`` for normalization by the sum of the target class weights present in the batch. Its ``normalize_distance=True`` option normalizes each sample's class distance weights to sum to one. This changes the relative penalties for target classes with different total class distances. The two controls can be used independently.

.. code-block:: python

    from dlordinal.losses import CDWCELoss

    criterion = CDWCELoss(
        num_classes=4,
        weight=weights,
        reduction="weighted_mean",
        normalize_distance=True,
    )
    loss = criterion(logits, targets)

The existing ``CDWCELoss`` defaults preserve its previous mean reduction and distance weights. ``NormalizedCrossEntropyLoss`` and ``CDWCELoss(reduction="weighted_mean")`` return zero with zero input gradients when the total effective weight is zero. ``NormalizedCrossEntropyLoss`` also supports PyTorch's ``none`` and ``sum`` reductions. ``CDWCELoss(reduction="none")`` returns one loss per sample, and ``reduction="sum"`` sums these losses.

.. automodule:: dlordinal.losses
    :members:

.. footbibliography::
