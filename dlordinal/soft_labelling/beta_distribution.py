import numpy as np
from scipy.special import gamma, hyp2f1

from .utils import get_intervals


def beta_func(a, b):
    """Compute the beta function.

    Parameters
    ----------
    a : float
            First parameter.
    b : float
            Second parameter.

    Returns
    -------
    beta_func : float
            The value of the beta function.
    """

    if a <= 0:
        raise ValueError(f"{a=} can not be negative")

    if b <= 0:
        raise ValueError(f"{b=} can not be negative")

    return (gamma(a) * gamma(b)) / gamma(a + b)


def beta_dist(x, p, q, a=1.0):
    """Compute the beta distribution function for value x with parameters p, q and a.

    Parameters
    ----------
    x : float
            Value to compute the distribution function for.
    p : float
            First shape parameter (:math:`p > 0`).
    q : float
            Second shape parameter (:math:`q > 0`).
    a : float, default=1.0
            Scaling parameter (:math:`a > 0`).

    Returns
    -------
    beta: float
            The value of the beta distribution function.
    """

    if x < 0 or x > 1:
        raise ValueError(f"{x=} must be in the interval [0,1]")

    if a <= 0:
        raise ValueError(f"{a=} can not be negative or 0")

    return (x ** (a * p)) / (p * beta_func(p, q)) * hyp2f1(p, 1 - q, p + 1, x**a)


def _get_beta_soft_label(J, p, q, a=1.0):
    """Get soft labels from a beta distribution :math:`B(p,q,a)` for ``J`` splits.
    The :math:`[0,1]` interval is split into ``J`` intervals and the probability for
    each interval is computed as the difference between the value of the distribution
    function in the upper limit of the interval and the value of the distribution
    function in the lower limit of the interval. Thus, the probability for class ``j``
    is computed as :math:`B(p,q,a)(j/J) - B(p,q,a)((j-1)/J)`.

    Parameters
    ----------
    J : int
            Number of classes or splits.
    p : float
            First shape parameter (:math:`p > 0`).
    q : float
            Second shape parameter (:math:`q > 0`).
    a : float, default=1.0
            Scaling parameter (:math:`a > 0`).

    Raises
    ------
    ValueError
            If ``J`` is not a positive integer, if ``p`` is not positive, if ``q`` is
            not positive or if ``a`` is not positive.

    Returns
    -------
    probs: list
            List of ``J`` elements that represent the probability associated with each
            class or split.

    Example
    -------
    >>> from dlordinal.soft_labelling import get_beta_soft_labels
    >>> get_beta_soft_labels(3, 2, 3)
    [0.4074074080000002, 0.48148148059259255, 0.11111111140740726]
    >>> get_beta_soft_labels(5, 5, 1, a=2)
    [1.0240000307200007e-07, 0.00010475520052121611, 0.005941759979320316,
      0.10132756401484902, 0.8926258084053067]
    """

    if J < 2 or not isinstance(J, int):
        raise ValueError(f"{J=} must be a positive integer greater than 1")

    if p <= 0:
        raise ValueError(f"{p=} must be positive")

    if q <= 0:
        raise ValueError(f"{q=} must be positive")

    if a <= 0:
        raise ValueError(f"{a=} must be positive")

    intervals = get_intervals(J)
    probs = []

    # Compute probability for each interval (class) using the distribution function.
    for interval in intervals:
        probs.append(beta_dist(interval[1], p, q, a) - beta_dist(interval[0], p, q, a))

    return probs


# Params [p,q,a] for beta distribution
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


def get_beta_soft_labels(J, params_set="standard"):
    """Get soft labels for each of the ``J`` classes using a beta distributions and
    the parameter defined in the ``params_set`` as described in :footcite:t:`vargas2022unimodal`.

    Parameters
    ----------
    J : int
            Number of classes or splits.

    params_set : str, default='standard'
            The set of parameters of the beta distributions employed to generate the
            soft labels. It has to be one of the keys in the ``_beta_params_sets``
            dictionary.

    Raises
    ------
    ValueError
            If ``J`` is not a positive integer or if ``params_set`` is not a valid key.

    Returns
    -------
    probs: list
            List of ``J`` elements where each elements is also a list of ``J`` elements.
            Each inner list represents the soft label of class ``j`` for each of the
            ``J`` classes. For example: ``probs[j]`` is the soft label for class ``j``.
            Then, ``probs[j][k]`` is the probability of assigning class ``k`` to the
            instance that belongs to class ``j``.

    Example
    -------
    >>> from dlordinal.soft_labelling import get_beta_softlabels
    >>> get_beta_softlabels(3)
    [[0.802469132197531, 0.18518518474074064, 0.01234567906172801],
    [0.17329675405578432, 0.6534064918884313, 0.1732967540557846],
    [0.012345679061728405, 0.1851851847407408, 0.8024691321975309]]
    >>> get_beta_softlabels(5)
    [[0.8322278330066323, 0.15097599903815717, 0.01614079995258888,
    0.0006528000025611824, 2.5600000609360407e-06], [0.16306230596685573,
    0.6740152119811103, 0.15985465217901373, 0.003067150177798128,
    6.796952229937148e-07], [0.0005973937258183486, 0.16304604740785777,
    0.6727131177326486, 0.16304604740785367, 0.000597393725820794],
    [6.796952207410873e-07, 0.0030671501777993896, 0.15985465217901168,
    0.6740152119811109, 0.16306230596685678], [2.560000061440001e-06,
    0.0006528000025600005, 0.01614079995258879, 0.1509759990381568,
    0.8322278330066332]]
    """

    if J <= 0:
        raise ValueError(f"{J=} must be a positive integer")

    if params_set not in _beta_params_sets:
        raise ValueError(f"Invalid params_set: {params_set}")

    params = _beta_params_sets[params_set]
    return np.array([_get_beta_soft_label(J, p, q, a) for (p, q, a) in params[J]])
