import numpy as np
from scipy.special import softmax
from scipy.stats import poisson


def get_poisson_soft_labels(J):
    """Get soft labels using poisson distributions for ``J`` classes or splits using the
    methodology described in :footcite:t:`liu2020unimodal`.

    For the target class :math:`j` (:math:`j = 0, \\ldots, J-1`), the poisson
    probability mass function :math:`\\text{p}(k; \\lambda)` with
    :math:`\\lambda = j + 1` is evaluated at :math:`k = 0, \\ldots, J-1`. Then, a
    softmax function is applied to each row of the resulting matrix to obtain valid
    probabilities, i.e. the soft label of class :math:`k` is proportional to
    :math:`\\exp(\\text{p}(k; j + 1))`.

    Note that the softmax is applied to the probability mass values themselves, not
    to their logarithms, so the resulting soft labels are much flatter than the
    poisson distributions. Moreover, because :math:`\\lambda = j + 1` is an integer,
    the probability mass function takes the same value at :math:`k = j` and
    :math:`k = j + 1`.

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
