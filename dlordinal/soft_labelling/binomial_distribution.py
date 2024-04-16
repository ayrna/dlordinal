import numpy as np
from scipy.stats import binom


def get_binomial_soft_labels(J):
    """Get soft labels for the binomial distribution for ``J`` classes or splits
    using the approach described in :footcite:t:`liu2020unimodal`.
    The :math:`[0,1]` interval is split into ``J`` intervals and the probability for
    each interval is computed as the difference between the value of the binomial
    probability function for the interval boundaries. The probability for the first
    interval is computed as the value of the binomial probability function for the first
    interval boundary.

    The binomial distributions employed are denoted as :math:`\\text{b}(k, n-1, p)` where
    :math:`k` is given by the order of the class for which the probability is computed,
    and :math:`p` is given by :math:`0.1 + (0.9-0.1) / (n-1) * j` where :math:`j` is
    is the order of the target class.

    Parameters
    ----------
    J : int
            Number of classes or splits.

    Raises
    ------
    ValueError
            If ``J`` is not a positive integer greater than 1.

    Returns
    -------
    probs : 2d array-like of shape (J, J)
            Matrix of probabilities where each row represents the true class
            and each column the probability for class ``j``.

    Example
    -------
    >>> from dlordinal.soft_labelling import get_binomial_soft_labels
    >>> get_binomial_soft_labels(5)
    array([[6.561e-01, 2.916e-01, 4.860e-02, 3.600e-03, 1.000e-04],
            [2.401e-01, 4.116e-01, 2.646e-01, 7.560e-02, 8.100e-03],
            [6.250e-02, 2.500e-01, 3.750e-01, 2.500e-01, 6.250e-02],
            [8.100e-03, 7.560e-02, 2.646e-01, 4.116e-01, 2.401e-01],
            [1.000e-04, 3.600e-03, 4.860e-02, 2.916e-01, 6.561e-01]])
    """

    if J < 2 or not isinstance(J, int):
        raise ValueError(f"{J=} must be a positive integer greater than 1")

    params = {}

    params["4"] = np.linspace(0.1, 0.9, 4)
    params["5"] = np.linspace(0.1, 0.9, 5)
    params["6"] = np.linspace(0.1, 0.9, 6)
    params["7"] = np.linspace(0.1, 0.9, 7)
    params["8"] = np.linspace(0.1, 0.9, 8)
    params["10"] = np.linspace(0.1, 0.9, 10)
    params["12"] = np.linspace(0.1, 0.9, 12)
    params["14"] = np.linspace(0.1, 0.9, 14)

    probs = []

    for true_class in range(0, J):
        probs.append(binom.pmf(np.arange(0, J), J - 1, params[str(J)][true_class]))

    return np.array(probs)
