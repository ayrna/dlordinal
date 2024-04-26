import numpy as np
from scipy.special import softmax


def get_exponential_soft_labels(J, p=1.0, tau=1.0):
    """Get soft labels from exponential distribution for ``J`` classes or splits as
    described in :footcite:t:`liu2020unimodal` and :footcite:t:`vargas2023exponential`.
    The :math:`[0,1]` interval is split into ``J`` intervals and the probability for
    each interval is computed as the difference between the value of the exponential
    function for the interval boundaries. The probability for the first interval is
    computed as the value of the exponential function for the first interval boundary.
    Then, a softmax function is applied to each row of the resulting matrix to obtain
    valid probabilities.

    The aforementioned exponential function is defined as follows:
    :math:`f(x; k, p, \\tau) = \\frac{-|x - k|^p}{\\tau}` where :math:`k` is given by the order of the
    true class for which the probability is computed, :math:`p` is the exponent
    parameter and :math:`\\tau` is the scaling parameter.

    Parameters
    ----------
    J : int
            Number of classes.
    p : float, default=1.0
            Exponent parameter :math:`p`.
    tau : float, default=1.0
            Scaling parameter :math:`\\tau`.

    Raises
    ------
    ValueError
            If ``J`` is not a positive integer greater than 1.

    Returns
    -------
    probs : 2d array-like of shape (J, J)
            Matrix of probabilities where each row represents the true class
            and each column the probability for class j.

    Example
    -------
    >>> from dlordinal.soft_labelling import get_exponential_soft_labels
    >>> get_exponential_soft_labels(5)
    array([[0.63640865, 0.23412166, 0.08612854, 0.03168492, 0.01165623],
    [0.19151597, 0.52059439, 0.19151597, 0.07045479, 0.02591887],
    [0.06745081, 0.1833503 , 0.49839779, 0.1833503 , 0.06745081],
    [0.02591887, 0.07045479, 0.19151597, 0.52059439, 0.19151597],
    [0.01165623, 0.03168492, 0.08612854, 0.23412166, 0.63640865]])
    """

    if J < 2 or not isinstance(J, int):
        raise ValueError(f"{J=} must be a positive integer greater than 1")

    probs = []

    for true_class in range(0, J):
        probs.append(-(np.abs(np.arange(0, J) - true_class) ** p) / tau)

    return softmax(np.array(probs), axis=1)
