import numpy as np
from scipy.special import gamma, hyp2f1

from .utils import get_intervals


def beta_inc(a, b):
    """Compute the incomplete beta function.

    Parameters
    ----------
    a : float
            First parameter.
    b : float
            Second parameter.

    Returns
    -------
    beta_inc: float
            The value of the incomplete beta function.
    """

    return (gamma(a) * gamma(b)) / gamma(a + b)


def beta(x, p, q, a=1.0):
    """Compute the beta density function for value x with parameters p, q and a.

    Parameters
    ----------
    x : float
            Value to compute the density function for.
    p: float
            First parameter of the beta distribution.
    q: float
            Second parameter of the beta distribution.
    a: float, default=1.0
            Scaling parameter.

    Returns
    -------
    beta: float
            The value of the beta density function.
    """

    return (
        gamma(p + q)
        / (gamma(p) * gamma(q))
        * a
        * x ** (a * p - 1)
        * (1 - x**a) ** (q - 1)
    )


def beta_dist(x, p, q, a=1.0):
    """Compute the beta distribution function for value x with parameters p, q and a.

    Parameters
    ----------
    x : float
            Value to compute the distribution function for.
    p: float
            First parameter of the beta distribution.
    q: float
            Second parameter of the beta distribution.
    a: float, default=1.0
            Scaling parameter.

    Returns
    -------
    beta: float
            The value of the beta distribution function.
    """

    return (x ** (a * p)) / (p * beta_inc(p, q)) * hyp2f1(p, 1 - q, p + 1, x**a)


def get_beta_probabilities(n, p, q, a=1.0):
    """Get probabilities from beta distribution (p,q,a) for n splits.

    Parameters
    ----------
    n : int
            Number of classes.
    p: float
            First parameter of the beta distribution.
    q: float
            Second parameter of the beta distribution.
    a: float, default=1.0
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
