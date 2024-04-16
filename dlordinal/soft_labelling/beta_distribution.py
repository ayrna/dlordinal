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


def get_beta_soft_labels(J, p, q, a=1.0):
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
