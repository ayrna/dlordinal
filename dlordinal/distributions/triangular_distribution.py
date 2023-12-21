import math

import numpy as np

from .utils import get_intervals, triangular_cdf


def get_triangular_probabilities(n: int, alpha2: float = 0.01, verbose: int = 0):
    """
    Get the probabilities for the triangular distribution.

    Parameters
    ----------
    n : int
        Number of classes.
    alpha2 : float, optional
        Alpha2 value, by default 0.01.
    verbose : int, optional
        Verbosity level, by default 0.
    """

    if verbose >= 1:
        print(f"Computing triangular probabilities for {n=} and {alpha2=}...")

    def compute_alpha1(alpha2):
        c_minus = (1 - 2 * alpha2) * (2 * alpha2 - math.sqrt(2 * alpha2))

        return pow((1 - math.sqrt(1 - 4 * c_minus)) / 2, 2)

    def compute_alpha3(alpha2):
        c1 = (
            pow((n - 1) / n, 2)
            * (1 - 2 * alpha2)
            * (math.sqrt(2 * alpha2) * (-1 + math.sqrt(2 * alpha2)))
        )

        return pow((1 - math.sqrt(1 - 4 * c1)) / 2, 2)

    alpha1 = compute_alpha1(alpha2)
    alpha3 = compute_alpha3(alpha2)

    if verbose >= 1:
        print(f"{alpha1=}, {alpha2=}, {alpha3=}")

    def b1(n):
        return 1.0 / ((1.0 - math.sqrt(alpha1)) * n)

    def m1(n):
        return math.sqrt(alpha1) / ((1 - math.sqrt(alpha1)) * n)

    def aj(n, j):
        num1 = 2.0 * j - 2 - 4 * j * alpha2 + 2 * alpha2
        num2 = math.sqrt(2 * alpha2)
        den = 2.0 * n * (1 - 2 * alpha2)

        max_value = (j - 1.0) / n

        # +-
        return (
            (num1 + num2) / den
            if (num1 + num2) / den < max_value
            else (num1 - num2) / den
        )

    def bj(n, j):
        num1 = 2.0 * j - 4 * j * alpha2 + 2 * alpha2
        num2 = math.sqrt(2 * alpha2)
        den = 2.0 * n * (1 - 2 * alpha2)

        min_value = j / n

        # +-
        return (
            (num1 + num2) / den
            if (num1 + num2) / den > min_value
            else (num1 - num2) / den
        )

    def aJ(n):
        aJ_plus = 1.0 + 1.0 / (n * (math.sqrt(alpha3) - 1.0))
        aJ_minus = 1.0 + 1.0 / (-n * (math.sqrt(alpha3) - 1.0))
        return aJ_plus if aJ_plus > 0.0 else aJ_minus

    def nJ(n):
        num = math.sqrt(alpha3)
        den = n * (1 - alpha3)
        return num / den

    if verbose >= 3:
        print(f"{b1(n)=}, {m1(n)=}, {aJ(n)=}, {nJ(n)=}, {aj(n, 1)=}, {bj(n,1)=}")
        for i in range(1, n + 1):
            print(f"{i=}  {aj(n, i)=}, {bj(n,i)=}")

    intervals = get_intervals(n)
    probs = []

    # Compute probability for each interval (class) using the distribution function.
    for j in range(1, n + 1):
        j_probs = []
        if j == 1:
            a = 0.0
            b = b1(n)
            c = 0.0
        elif j == n:
            a = aJ(n)
            b = 1.0
            c = 1.0
        else:
            a = aj(n, j)
            b = bj(n, j)
            c = (a + b) / 2.0

        if verbose >= 1:
            print(f"Class: {j}, {a=}, {b=}, {c=}, (j-1)/J={(j-1)/n}, (j/J)={j/n}")

        for interval in intervals:
            j_probs.append(
                triangular_cdf(interval[1], a, b, c)
                - triangular_cdf(interval[0], a, b, c)
            )
            if verbose >= 2:
                print(f"\tinterval: {interval}, prob={j_probs[-1]}")

        probs.append(j_probs)

    return np.array(probs)
