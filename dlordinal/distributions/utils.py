import numpy as np


def get_intervals(n):
    """Get n evenly-spaced intervals in :math:`[0,1]`.

    Parameters
    ----------
    n : int
        Number of intervals.

    Returns
    -------
    intervals: list
        List of intervals.
    """

    points = np.linspace(1e-9, 1 - 1e-9, n + 1)
    intervals = []
    for i in range(0, points.size - 1):
        intervals.append((points[i], points[i + 1]))

    return intervals


def triangular_cdf(x: float, a: float, b: float, c: float):
    """
    Triangular distribution CDF.

    Parameters
    ----------
    x : float
        Value.
    a : float
        Lower bound.
    b : float
        Upper bound.
    c : float
        Mode.
    """
    if x <= a:
        return 0
    elif a < x < c:
        return pow(x - a, 2) / ((b - a) * (c - a))
    elif c < x < b:
        return 1 - pow(b - x, 2) / ((b - a) * (b - c))
    else:  # b <= x
        return 1
