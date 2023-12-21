from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from torch.nn.modules.loss import MSELoss

from ..distributions import (
    get_beta_probabilities,
    get_binomial_probabilities,
    get_exponential_probabilities,
    get_general_triangular_probabilities,
    get_poisson_probabilities,
    get_triangular_probabilities,
)
from .wkloss import WKLoss

# Params [a,b] for beta distribution
_beta_params_sets = {
    "standard": {
        3: [[1, 4, 1], [4, 4, 1], [4, 1, 1]],
        4: [[1, 6, 1], [6, 10, 1], [10, 6, 1], [6, 1, 1]],
        5: [[1, 8, 1], [6, 14, 1], [12, 12, 1], [14, 6, 1], [8, 1, 1]],
        6: [[1, 10, 1], [7, 20, 1], [15, 20, 1], [20, 15, 1], [20, 7, 1], [10, 1, 1]],
        7: [
            [1, 12, 1],
            [7, 26, 1],
            [16, 28, 1],
            [24, 24, 1],
            [28, 16, 1],
            [26, 7, 1],
            [12, 1, 1],
        ],
        8: [
            [1, 14, 1],
            [7, 31, 1],
            [17, 37, 1],
            [27, 35, 1],
            [35, 27, 1],
            [37, 17, 1],
            [31, 7, 1],
            [14, 1, 1],
        ],
        9: [
            [1, 16, 1],
            [8, 40, 1],
            [18, 47, 1],
            [30, 47, 1],
            [40, 40, 1],
            [47, 30, 1],
            [47, 18, 1],
            [40, 8, 1],
            [16, 1, 1],
        ],
        10: [
            [1, 18, 1],
            [8, 45, 1],
            [19, 57, 1],
            [32, 59, 1],
            [45, 55, 1],
            [55, 45, 1],
            [59, 32, 1],
            [57, 19, 1],
            [45, 8, 1],
            [18, 1, 1],
        ],
        11: [
            [1, 21, 1],
            [8, 51, 1],
            [20, 68, 1],
            [34, 73, 1],
            [48, 69, 1],
            [60, 60, 1],
            [69, 48, 1],
            [73, 34, 1],
            [68, 20, 1],
            [51, 8, 1],
            [21, 1, 1],
        ],
        12: [
            [1, 23, 1],
            [8, 56, 1],
            [20, 76, 1],
            [35, 85, 1],
            [51, 85, 1],
            [65, 77, 1],
            [77, 65, 1],
            [85, 51, 1],
            [85, 35, 1],
            [76, 20, 1],
            [56, 8, 1],
            [23, 1, 1],
        ],
        13: [
            [1, 25, 1],
            [8, 61, 1],
            [20, 84, 1],
            [36, 98, 1],
            [53, 100, 1],
            [70, 95, 1],
            [84, 84, 1],
            [95, 70, 1],
            [100, 53, 1],
            [98, 36, 1],
            [84, 20, 1],
            [61, 8, 1],
            [25, 1, 1],
        ],
        14: [
            [1, 27, 1],
            [2, 17, 1],
            [5, 23, 1],
            [9, 27, 1],
            [13, 28, 1],
            [18, 28, 1],
            [23, 27, 1],
            [27, 23, 1],
            [28, 18, 1],
            [28, 13, 1],
            [27, 9, 1],
            [23, 5, 1],
            [17, 2, 1],
            [27, 1, 1],
        ],
    }
}


class CustomTargetsCrossEntropyLoss(torch.nn.Module):
    """Base class to implement a unimodal regularised cross entropy loss.
    Vargas, V. M., Gutiérrez, P. A., & Hervás-Martínez, C. (2022).
    Unimodal regularisation based on beta distribution for deep ordinal regression.
    Pattern Recognition, 122, 108310.

    Parameters
    ----------
    cls_probs : Tensor
        The class probabilities tensor.
    eta : float, default=1.0
        Parameter that controls the influence of the regularisation.
    weight : Optional[Tensor], default=None
        A manual rescaling weight given to each class. If given, has to be a Tensor
        of size `C`. Otherwise, it is treated as if having all ones.
    size_average : Optional[bool], default=None
        Deprecated (see :attr:`reduction`). By default, the losses are averaged over
        each loss element in the batch. Note that for some losses, there are
        multiple elements per sample. If the field :attr:`size_average` is set to
        ``False``, the losses are instead summed for each minibatch. Ignored when
        reduce is ``False``. Default: ``True``
    ignore_index : int, default=-100
        Specifies a target value that is ignored and does not contribute to the
        input gradient. When :attr:`size_average` is ``True``, the loss is averaged
        over non-ignored targets.
    reduce : Optional[bool], default=None
        Deprecated (see :attr:`reduction`). By default, the losses are averaged or
        summed over observations for each minibatch depending on :attr:`size_average`.
        When :attr:`reduce` is ``False``, returns a loss per batch element instead
        and ignores :attr:`size_average`. Default: ``True``
    reduction : str, default='mean'
        Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` |
        ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of
        the output will be divided by the number of elements in the output,
        ``'sum'``: the output will be summed. Note: :attr:`size_average` and
        :attr:`reduce` are in the process of being deprecated, and in the meantime,
        specifying either of those two args will override :attr:`reduction`.
        Default: ``'mean'``
    label_smoothing : float, default=0.0
        Controls the amount of label smoothing for the loss. Zero means no smoothing.
        Default: ``0.0``
    """

    def __init__(
        self,
        cls_probs: Tensor,
        eta: float = 1.0,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.num_classes = cls_probs.size(0)
        self.eta = eta

        self.ce_loss = CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

        # Default class probs initialized to ones
        self.register_buffer("cls_probs", cls_probs.float())

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Method that is called to compute the loss.

        Parameters
        ----------
        input : Tensor
            The input tensor.
        target : Tensor
            The target tensor.

        Returns
        -------
        loss: Tensor
            The computed loss.
        """

        y_prob = self.get_buffer("cls_probs")[target]
        target_oh = one_hot(target, self.num_classes)

        y_true = (1.0 - self.eta) * target_oh + self.eta * y_prob

        return self.ce_loss(input, y_true)


class BetaCrossEntropyLoss(CustomTargetsCrossEntropyLoss):
    """Beta unimodal regularised cross entropy loss.
    Vargas, Víctor Manuel et al. (2022). Unimodal regularisation based on beta
    distribution for deep ordinal regression. Pattern Recognition, 122, 108310.
    Elsevier.

    Parameters
    ----------
    num_classes : int, default=5
        Number of classes.
    params_set : str, default='standard'
        The set of parameters to use for the beta distribution (chosen from the
        _beta_params_set dictionary).
    eta : float, default=1.0
        Parameter that controls the influence of the regularisation.
    weight : Optional[Tensor], default=None
        A manual rescaling weight given to each class. If given, has to be a Tensor
        of size `C`. Otherwise, it is treated as if having all ones.
    size_average : Optional[bool], default=None
        Deprecated (see :attr:`reduction`). By default, the losses are averaged over
        each loss element in the batch. Note that for some losses, there are
        multiple elements per sample. If the field :attr:`size_average` is set to
        ``False``, the losses are instead summed for each minibatch. Ignored when
        reduce is ``False``. Default: ``True``
    ignore_index : int, default=-100
        Specifies a target value that is ignored and does not contribute to the
        input gradient. When :attr:`size_average` is ``True``, the loss is averaged
        over non-ignored targets.
    reduce : Optional[bool], default=None
        Deprecated (see :attr:`reduction`). By default, the losses are averaged or
        summed over observations for each minibatch depending on :attr:`size_average`.
        When :attr:`reduce` is ``False``, returns a loss per batch element instead
        and ignores :attr:`size_average`. Default: ``True``
    reduction : str, default='mean'
        Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` |
        ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of
        the output will be divided by the number of elements in the output,
        ``'sum'``: the output will be summed. Note: :attr:`size_average` and
        :attr:`reduce` are in the process of being deprecated, and in the meantime,
        specifying either of those two args will override :attr:`reduction`.
        Default: ``'mean'``
    label_smoothing : float, default=0.0
        Controls the amount of label smoothing for the loss. Zero means no smoothing.
        Default: ``0.0``
    """

    def __init__(
        self,
        num_classes: int = 5,
        params_set: str = "standard",
        eta: float = 1.0,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        self.params = _beta_params_sets[params_set]

        # Precompute class probabilities for each label
        cls_probs = torch.tensor(
            [
                get_beta_probabilities(
                    num_classes,
                    self.params[num_classes][i][0],
                    self.params[num_classes][i][1],
                    self.params[num_classes][i][2],
                )
                for i in range(num_classes)
            ]
        ).float()

        super().__init__(
            cls_probs=cls_probs,
            eta=eta,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )


class TriangularCrossEntropyLoss(CustomTargetsCrossEntropyLoss):
    """Triangular unimodal regularised cross entropy loss.
    Víctor Manuel Vargas, Pedro Antonio Gutiérrez, Javier Barbero-Gómez, and
    César Hervás-Martínez (2023). Soft Labelling Based on Triangular Distributions for
    Ordinal Classification. Information Fusion, 93, 258--267.

    Parameters
    ----------
    num_classes : int, default=5
        Number of classes.
    alpha2 : float, default=0.05
        Parameter that controls the influence of the regularisation.
    eta : float, default=1.0
        Parameter that controls the influence of the regularisation.
    weight : Optional[Tensor], default=None
        A manual rescaling weight given to each class. If given, has to be a Tensor
        of size `C`. Otherwise, it is treated as if having all ones.
    size_average : Optional[bool], default=None
        Deprecated (see :attr:`reduction`). By default, the losses are averaged over
        each loss element in the batch. Note that for some losses, there are
        multiple elements per sample. If the field :attr:`size_average` is set to
        ``False``, the losses are instead summed for each minibatch. Ignored when
        reduce is ``False``. Default: ``True``
    ignore_index : int, default=-100
        Specifies a target value that is ignored and does not contribute to the
        input gradient. When :attr:`size_average` is ``True``, the loss is averaged
        over non-ignored targets.
    reduce : Optional[bool], default=None
        Deprecated (see :attr:`reduction`). By default, the losses are averaged or
        summed over observations for each minibatch depending on :attr:`size_average`.
        When :attr:`reduce` is ``False``, returns a loss per batch element instead
        and ignores :attr:`size_average`. Default: ``True``
    reduction : str, default='mean'
        Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` |
        ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of
        the output will be divided by the number of elements in the output,
        ``'sum'``: the output will be summed. Note: :attr:`size_average` and
        :attr:`reduce` are in the process of being deprecated, and in the meantime,
        specifying either of those two args will override :attr:`reduction`.
        Default: ``'mean'``
    label_smoothing : float, default=0.0
        Controls the amount of label smoothing for the loss. Zero means no smoothing.
        Default: ``0.0``
    """

    def __init__(
        self,
        num_classes: int = 5,
        alpha2: float = 0.05,
        eta: float = 1.0,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        # Precompute class probabilities for each label
        cls_probs = torch.tensor(get_triangular_probabilities(num_classes, alpha2))
        super().__init__(
            cls_probs=cls_probs,
            eta=eta,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )


class GeneralTriangularCrossEntropyLoss(CustomTargetsCrossEntropyLoss):
    """Generalised triangular unimodal regularised cross entropy loss.
    Víctor Manuel Vargas, Antonio Manuel Durán-Rosal, David Guijo-Rubio,
    Pedro Antonio Gutiérrez-Peña, and César Hervás-Martínez (2023). Generalised
    Triangular Distributions for ordinal deep learning: novel proposal and
    optimisation. Information Sciences, 648, 1--17.

    Parameters
    ----------
    num_classes : int
        Number of classes.
    alphas : np.ndarray
        The alpha parameters for the triangular distribution.
    eta : float, default=1.0
        Parameter that controls the influence of the regularisation.
    weight : Optional[Tensor], default=None
        A manual rescaling weight given to each class. If given, has to be a Tensor
        of size `C`. Otherwise, it is treated as if having all ones.
    size_average : Optional[bool], default=None
        Deprecated (see :attr:`reduction`). By default, the losses are averaged over
        each loss element in the batch. Note that for some losses, there are
        multiple elements per sample. If the field :attr:`size_average` is set to
        ``False``, the losses are instead summed for each minibatch. Ignored when
        reduce is ``False``. Default: ``True``
    ignore_index : int, default=-100
        Specifies a target value that is ignored and does not contribute to the
        input gradient. When :attr:`size_average` is ``True``, the loss is averaged
        over non-ignored targets.
    reduce : Optional[bool], default=None
        Deprecated (see :attr:`reduction`). By default, the losses are averaged or
        summed over observations for each minibatch depending on :attr:`size_average`.
        When :attr:`reduce` is ``False``, returns a loss per batch element instead
        and ignores :attr:`size_average`. Default: ``True``
    reduction : str, default='mean'
        Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` |
        ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of
        the output will be divided by the number of elements in the output,
        ``'sum'``: the output will be summed. Note: :attr:`size_average` and
        :attr:`reduce` are in the process of being deprecated, and in the meantime,
        specifying either of those two args will override :attr:`reduction`.
        Default: ``'mean'``
    label_smoothing : float, default=0.0
        Controls the amount of label smoothing for the loss. Zero means no smoothing.
        Default: ``0.0``
    """

    def __init__(
        self,
        num_classes: int,
        alphas: np.ndarray,
        eta: float = 1.0,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        # Precompute class probabilities for each label
        r = get_general_triangular_probabilities(num_classes, alphas, verbose=3)
        cls_probs = torch.tensor(r)

        super().__init__(
            cls_probs=cls_probs,
            eta=eta,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )


class ExponentialRegularisedCrossEntropyLoss(CustomTargetsCrossEntropyLoss):
    """Expontential unimodal regularised cross entropy loss.
    Vargas, Víctor Manuel et al. (2022). Unimodal regularisation based on beta
    distribution for deep ordinal regression. Pattern Recognition, 122, 108310.
    Elsevier.

    Parameters
    ----------
    num_classes : int, default=5
        Number of classes.
    eta : float, default=1.0
        Parameter that controls the influence of the regularisation.
    weight : Optional[Tensor], default=None
        A manual rescaling weight given to each class. If given, has to be a Tensor
        of size `C`. Otherwise, it is treated as if having all ones.
    size_average : Optional[bool], default=None
        Deprecated (see :attr:`reduction`). By default, the losses are averaged over
        each loss element in the batch. Note that for some losses, there are
        multiple elements per sample. If the field :attr:`size_average` is set to
        ``False``, the losses are instead summed for each minibatch. Ignored when
        reduce is ``False``. Default: ``True``
    ignore_index : int, default=-100
        Specifies a target value that is ignored and does not contribute to the
        input gradient. When :attr:`size_average` is ``True``, the loss is averaged
        over non-ignored targets.
    reduce : Optional[bool], default=None
        Deprecated (see :attr:`reduction`). By default, the losses are averaged or
        summed over observations for each minibatch depending on :attr:`size_average`.
        When :attr:`reduce` is ``False``, returns a loss per batch element instead
        and ignores :attr:`size_average`. Default: ``True``
    reduction : str, default='mean'
        Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` |
        ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of
        the output will be divided by the number of elements in the output,
        ``'sum'``: the output will be summed. Note: :attr:`size_average` and
        :attr:`reduce` are in the process of being deprecated, and in the meantime,
        specifying either of those two args will override :attr:`reduction`.
        Default: ``'mean'``
    label_smoothing : float, default=0.0
        Controls the amount of label smoothing for the loss. Zero means no smoothing.
        Default: ``0.0``
    """

    def __init__(
        self,
        num_classes: int = 5,
        eta: float = 1.0,
        p: float = 1,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        # Precompute class probabilities for each label
        cls_probs = torch.tensor(get_exponential_probabilities(num_classes, p)).float()

        super().__init__(
            cls_probs=cls_probs,
            eta=eta,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )


class BinomialCrossEntropyLoss(CustomTargetsCrossEntropyLoss):
    """Binomial unimodal regularised cross entropy loss.
    Vargas, Víctor Manuel, et al. (2023). Exponential loss regularisation for
    encouraging ordinal constraint to shotgun stocks quality assessment. Applied Soft
    Computing, 138, 110191.

    Parameters
    ----------
    num_classes : int, default=5
        Number of classes.
    eta : float, default=1.0
        Parameter that controls the influence of the regularisation.
    weight : Optional[Tensor], default=None
        A manual rescaling weight given to each class. If given, has to be a Tensor
        of size `C`. Otherwise, it is treated as if having all ones.
    size_average : Optional[bool], default=None
        Deprecated (see :attr:`reduction`). By default, the losses are averaged over
        each loss element in the batch. Note that for some losses, there are
        multiple elements per sample. If the field :attr:`size_average` is set to
        ``False``, the losses are instead summed for each minibatch. Ignored when
        reduce is ``False``. Default: ``True``
    ignore_index : int, default=-100
        Specifies a target value that is ignored and does not contribute to the
        input gradient. When :attr:`size_average` is ``True``, the loss is averaged
        over non-ignored targets.
    reduce : Optional[bool], default=None
        Deprecated (see :attr:`reduction`). By default, the losses are averaged or
        summed over observations for each minibatch depending on :attr:`size_average`.
        When :attr:`reduce` is ``False``, returns a loss per batch element instead
        and ignores :attr:`size_average`. Default: ``True``
    reduction : str, default='mean'
        Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` |
        ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of
        the output will be divided by the number of elements in the output,
        ``'sum'``: the output will be summed. Note: :attr:`size_average` and
        :attr:`reduce` are in the process of being deprecated, and in the meantime,
        specifying either of those two args will override :attr:`reduction`.
        Default: ``'mean'``
    label_smoothing : float, default=0.0
        Controls the amount of label smoothing for the loss. Zero means no smoothing.
        Default: ``0.0``
    """

    def __init__(
        self,
        num_classes: int = 5,
        eta: float = 1.0,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        # Precompute class probabilities for each label
        cls_probs = torch.tensor(get_binomial_probabilities(num_classes)).float()

        super().__init__(
            cls_probs=cls_probs,
            eta=eta,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )


class PoissonCrossEntropyLoss(CustomTargetsCrossEntropyLoss):
    """Poisson unimodal regularised cross entropy loss
    Liu, Xiaofeng et al. (2020). Unimodal regularized neuron stick-breaking for
    ordinal classification. Neurocomputing, 388, 34-44.

    Parameters
    ----------
    num_classes : int, default=5
        Number of classes.
    eta : float, default=1.0
        Parameter that controls the influence of the regularisation.
    weight : Optional[Tensor], default=None
        A manual rescaling weight given to each class. If given, has to be a Tensor
        of size `C`. Otherwise, it is treated as if having all ones.
    size_average : Optional[bool], default=None
        Deprecated (see :attr:`reduction`). By default, the losses are averaged over
        each loss element in the batch. Note that for some losses, there are
        multiple elements per sample. If the field :attr:`size_average` is set to
        ``False``, the losses are instead summed for each minibatch. Ignored when
        reduce is ``False``. Default: ``True``
    ignore_index : int, default=-100
        Specifies a target value that is ignored and does not contribute to the
        input gradient. When :attr:`size_average` is ``True``, the loss is averaged
        over non-ignored targets.
    reduce : Optional[bool], default=None
        Deprecated (see :attr:`reduction`). By default, the losses are averaged or
        summed over observations for each minibatch depending on :attr:`size_average`.
        When :attr:`reduce` is ``False``, returns a loss per batch element instead
        and ignores :attr:`size_average`. Default: ``True``
    reduction : str, default='mean'
        Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` |
        ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of
        the output will be divided by the number of elements in the output,
        ``'sum'``: the output will be summed. Note: :attr:`size_average` and
        :attr:`reduce` are in the process of being deprecated, and in the meantime,
        specifying either of those two args will override :attr:`reduction`.
        Default: ``'mean'``
    label_smoothing : float, default=0.0
        Controls the amount of label smoothing for the loss. Zero means no smoothing.
        Default: ``0.0``
    """

    def __init__(
        self,
        num_classes: int = 5,
        eta: float = 1.0,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        # Precompute class probabilities for each label
        cls_probs = torch.tensor(get_poisson_probabilities(num_classes)).float()

        super().__init__(
            cls_probs=cls_probs,
            eta=eta,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )


# class WKLoss(torch.nn.Module):
#     """Weighted Kappa loss implementation.
#     de La Torre, J., Puig, D., & Valls, A. (2018).
#     Weighted kappa loss function for multi-class classification of ordinal data in deep
#     learning. Pattern Recognition Letters, 105, 144-154.

#     Parameters
#     ----------
#     num_classes : int
#         Number of classes.
#     penalization_type : str, default='quadratic'
#         The penalization type of WK loss to use (quadratic or linear).
#     weight : np.ndarray, default=None
#         The weight matrix that is applied to the cost matrix.
#     """

#     cost_matrix: Tensor

#     def __init__(
#         self,
#         num_classes: int,
#         penalization_type: str = "quadratic",
#         weight: Optional[Tensor] = None,
#     ) -> None:
#         super().__init__()

#         # Create cost matrix and register as buffer
#         cost_matrix = np.reshape(
#             np.tile(range(num_classes), num_classes), (num_classes, num_classes)
#         )

#         if penalization_type == "quadratic":
#             cost_matrix = (
#                 np.power(cost_matrix - np.transpose(cost_matrix), 2)
#                 / (num_classes - 1) ** 2.0
#             )
#         else:
#             cost_matrix = (
#                 np.abs(cost_matrix - np.transpose(cost_matrix))
#                 / (num_classes - 1) ** 2.0
#             )

#         self.weight = weight
#         if isinstance(weight, np.ndarray):
#             weight = torch.tensor(weight, dtype=torch.float)

#         cost_matrix = torch.tensor(cost_matrix, dtype=torch.float)

#         if self.weight is not None:
#             tiled_weight = torch.tile(self.weight, (num_classes, 1)).T
#             cost_matrix = cost_matrix * tiled_weight

#         self.register_buffer("cost_matrix", cost_matrix)

#         self.num_classes = num_classes

#     def forward(self, input: Tensor, target: Tensor) -> Tensor:
#         """
#         Parameters
#         ----------
#         input : torch.Tensor
#             The input tensor.
#         target : torch.Tensor
#             The target tensor.

#         Returns
#         -------
#         loss: Tensor
#             The WK loss.
#         """

#         input = torch.nn.functional.softmax(input, dim=1)

#         costs = self.cost_matrix[target]

#         numerator = costs * input
#         numerator = torch.sum(numerator)

#         sum_prob = torch.sum(input, dim=0)
#         target_prob = one_hot(target, self.num_classes)
#         n = torch.sum(target_prob, dim=0)

#         a = torch.reshape(
#             torch.matmul(self.cost_matrix, torch.reshape(sum_prob, shape=[-1, 1])),
#             shape=[-1],
#         )

#         b = torch.reshape(n / torch.sum(n), shape=[-1])

#         epsilon = 1e-9

#         denominator = a * b
#         denominator = torch.sum(denominator) + epsilon

#         result = numerator / denominator

#         return result


class MCELoss(torch.nn.modules.loss._WeightedLoss):
    """
    Mean squared error per class loss function. Computes the mean squared error for each
    class and reduces it using the specified `reduction`.

    Parameters
    ----------
    num_classes : int
        Number of classes.

    weight : Optional[Tensor], default=None
        A manual rescaling weight given to each class. If given, has to be a Tensor
        of size `J`, where `J` is the number of classes.
        Otherwise, it is treated as if having all ones.

    reduction : str, default='mean'
        Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` |
        ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of
        the output will be divided by the number of elements in the output,
        ``'sum'``: the output will be summed.
    """

    def __init__(
        self,
        num_classes: int,
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(
            weight=weight, size_average=None, reduce=None, reduction=reduction
        )

        self.num_classes = num_classes

        if weight is not None and weight.shape != (num_classes,):
            raise ValueError(
                f"Weight shape {weight.shape} is not compatible"
                + "with num_classes {num_classes}"
            )

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Reduction {reduction} is not supported."
                + " Please use 'mean', 'sum' or 'none'"
            )

    def compute_per_class_mse(self, input: torch.Tensor, target: torch.Tensor):
        """
        Computes the MSE for each class independently.

        Parameters
        ----------
        input : torch.Tensor
            Predicted labels
        target : torch.Tensor
            Ground truth labels

        Returns:
        mses : torch.Tensor
            MSE values
        """

        if input.shape != target.shape:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes)

        if input.shape != target.shape:
            raise ValueError(
                f"Input shape {input.shape} is not compatible with target shape "
                + f"{target.shape}"
            )

        # Compute the squared error for each class
        per_class_se = torch.pow(target - input, 2)

        # Apply class weights if defined
        if self.weight is not None:
            tiled_weight = torch.tile(self.weight, (per_class_se.shape[0], 1))
            per_class_se = per_class_se * tiled_weight

        # Compute the mean squared error for each class
        per_class_mse = torch.mean(per_class_se, dim=0)

        return per_class_mse

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Parameters
        ----------
        input : torch.Tensor
            Predicted labels
        target : torch.Tensor
            Ground truth labels

        Returns
        -------
        reduced_mse : torch.Tensor
            MSE per class reduced using the specified `reduction`. If reduction is `none`,
            then a tensor with the MSE for each class is returned.
        """

        input = torch.nn.functional.softmax(input, dim=1)
        target_oh = torch.nn.functional.one_hot(target, num_classes=self.num_classes)

        per_class_mse = self.compute_per_class_mse(input, target_oh)

        if self.reduction == "mean":
            reduced_mse = torch.mean(per_class_mse)
        elif self.reduction == "sum":
            reduced_mse = torch.sum(per_class_mse)
        else:
            reduced_mse = per_class_mse

        return reduced_mse


class MCEAndWKLoss(torch.nn.modules.loss._WeightedLoss):
    """
    The loss function integrates both MCELoss and WKLoss, concurrently minimising
    error distances while preventing the omission of classes from predictions.

    Parameters
    ----------
    num_classes : int
        Number of classes
    C: float, defaul=0.5
        Weights the WK loss (C) and the MCE loss (1-C). Must be between 0 and 1.
    wk_penalization_type : str, default='quadratic'
        The penalization type of WK loss to use (quadratic or linear).
        See WKLoss for more details.
    weight : Optional[Tensor], default=None
        A manual rescaling weight given to each class. If given, has to be a Tensor
        of size `J`, where `J` is the number of classes.
        Otherwise, it is treated as if having all ones.
    reduction : str, default='mean'
        Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` |
        ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of
        the output will be divided by the number of elements in the output,
        ``'sum'``: the output will be summed.
    """

    def __init__(
        self,
        num_classes: int,
        C: float = 0.5,
        wk_penalization_type: str = "quadratic",
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(
            weight=weight, size_average=None, reduce=None, reduction=reduction
        )

        self.num_classes = num_classes
        self.C = C
        self.wk_penalization_type = wk_penalization_type

        if weight is not None and weight.shape != (num_classes,):
            raise ValueError(
                f"Weight shape {weight.shape} is not compatible"
                + "with num_classes {num_classes}"
            )

        if C < 0 or C > 1:
            raise ValueError(f"C must be between 0 and 1, but is {C}")

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Reduction {reduction} is not supported."
                + " Please use 'mean', 'sum' or 'none'"
            )

        self.wk = WKLoss(
            self.num_classes,
            penalization_type=self.wk_penalization_type,
            weight=weight,
        )
        self.mce = MCELoss(self.num_classes, weight=weight)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        Parameters
        ----------
        y_true : torch.Tensor
            Ground truth labels
        y_pred : torch.Tensor
            Predicted labels

        Returns
        -------
        loss : torch.Tensor
            The weighted sum of MCE and QWK loss
        """

        wk_result = self.wk(y_true, y_pred)
        mce_result = self.mce(y_true, y_pred)

        return self.C * wk_result + (1 - self.C) * mce_result


class OrdinalEcocDistanceLoss(torch.nn.Module):
    """Ordinal ECOC distance loss implementation.
    Barbero-Gómez, J., Gutiérrez, P. A., & Hervás-Martínez, C. (2022). Error-correcting
    output codes in the framework of deep ordinal classification. Neural Processing
    Letters, 1-32. Springer.

    Parameters
    ----------
    num_classes : int
        Number of classes
    device : torch.device
        Contains the device on which the model is running
    class_weights : Optional[torch.Tensor]
        Contains the weights for each class
    """

    def __init__(
        self, num_classes: int, device, class_weights: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()

        self.target_class = np.ones((num_classes, num_classes - 1), dtype=np.float32)
        self.target_class[np.triu_indices(num_classes, 0, num_classes - 1)] = 0.0
        self.target_class = torch.tensor(
            self.target_class, dtype=torch.float32, device=device, requires_grad=False
        )

        self.class_weights = class_weights

        if self.class_weights is not None:
            assert self.class_weights.shape == (num_classes,)
            self.class_weights = self.class_weights.float().to(device)
            self.mse = MSELoss(reduction="none")
        else:
            self.mse = MSELoss(reduction="sum")

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Parameters
        ----------
        target : torch.Tensor
            Ground truth labels
        input : torch.Tensor
            Predicted labels

        Returns
        -------
        loss : torch.Tensor
            If class_weights is not None, the weighted sum of the MSE loss
            Else the sum of the MSE loss
        """

        target_indices = target.long()

        if self.class_weights is not None:
            target_vector = self.target_class[target_indices]
            weights = self.class_weights[target_indices]  # type: ignore
            return (self.mse(input, target_vector).sum(dim=1) * weights).sum()
        else:
            target_vector = self.target_class[target_indices]
            return self.mse(input, target_vector)
