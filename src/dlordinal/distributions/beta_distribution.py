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
            Scaling parameter.

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


def get_beta_probabilities(n, p, q, a=1.0):
    """Get probabilities from a beta distribution :math:`B(p,q,a)` for ``n`` splits.

    Parameters
    ----------
    n : int
            Number of classes.
    p : float
            First shape parameter (:math:`p > 0`).
    q : float
            Second shape parameter (:math:`q > 0`).
    a : float, default=1.0
            Scaling parameter.

    Returns
    -------
    probs: list
            List of probabilities.
    """

    intervals = get_intervals(n)
    probs = []

    # Compute probability for each interval (class) using the distribution function.
    for interval in intervals:
        probs.append(beta_dist(interval[1], p, q, a) - beta_dist(interval[0], p, q, a))

    return probs
