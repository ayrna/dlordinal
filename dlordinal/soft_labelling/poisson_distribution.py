import numpy as np
from scipy.special import softmax
from scipy.stats import poisson


def get_poisson_soft_labels(J):
    """Get soft labels using poisson distributions for ``J`` classes or splits using the
    methodology described in :footcite:t:`liu2020unimodal`.
    The :math:`[0,1]` interval is split into ``J`` intervals and the probability for
    each interval is computed as the difference between the value of the poisson
    probability function for the interval boundaries. The probability for the first
    interval is computed as the value of the poisson probability function for the first
    interval boundary. Then, a softmax function is applied to each row of the resulting
    matrix to obtain valid probabilities.

    The poisson probability function is denoted as :math:`\\text{p}(k, \\lambda)`
    where :math:`k` is given by the order of the class for which the probability is
    computed, and :math:`\\lambda` is given by :math:`k` where :math:`k` is the order
    of the target class.

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
        and each column the probability for class j.

    Example
    -------
    >>> from dlordinal.soft_labelling import get_poisson_soft_labels
    >>> get_poisson_soft_labels(5)
    array([[0.23414552, 0.23414552, 0.19480578, 0.17232403, 0.16457916],
        [0.18896888, 0.21635436, 0.21635436, 0.19768881, 0.18063359],
        [0.17822335, 0.19688341, 0.21214973, 0.21214973, 0.20059378],
        [0.17919028, 0.18931175, 0.20370191, 0.21389803, 0.21389803],
        [0.18400408, 0.18903075, 0.19882883, 0.21031236, 0.21782399]])

    """

    if J < 2 or not isinstance(J, int):
        raise ValueError(f"{J=} must be a positive integer greater than 1")

    probs = []

    for true_class in range(1, J + 1):
        probs.append(poisson.pmf(np.arange(0, J), true_class))

    return softmax(np.array(probs), axis=1)
