import math

import numpy as np
from sympy import Eq, nsolve, solve, symbols
from sympy.core.add import Add

from .utils import get_intervals, triangular_cdf


def get_general_triangular_params(n: int, alphas: np.ndarray, verbose: int = 0):
    """
    Get the parameters for the general triangular distribution.

    Parameters
    ----------
    n : int
        Number of classes.
    alphas : np.ndarray
        Array of alphas.
    verbose : int, optional
        Verbosity level, by default 0.
    """
    alphas = np.array(alphas)
    if not isinstance(alphas, np.ndarray) or alphas.shape != (2 * n,):
        raise ValueError(
            f"alphas must be a numpy array of shape (2 * n,), but got {alphas.shape}"
        )

    def abc1(n, alpha):
        a = 0
        b = 1.0 / ((1.0 - math.sqrt(alpha)) * n)
        c = 0
        return a, b, c

    def abcJ(n, alpha):
        a = 1.0 - 1.0 / (n * (1.0 - math.sqrt(alpha)))
        b = 1
        c = 1
        return a, b, c

    def abcj(n, j, alpha2j_1, alpha2j):
        # Calculation of the terms of the system of equations
        # b = ...
        numa2 = 2 * n**2 - 2 * n**2 * alpha2j_1
        numa1 = 2 * j * n * alpha2j_1 - n * alpha2j_1 - 4 * n * j + 4 * n
        numa0 = 2 * j**2 - 4 * j + 2
        dena0 = 2 * j * n * alpha2j_1 - n * alpha2j_1
        dena1 = -(2 * n**2 * alpha2j_1)

        # a = ...
        numb2 = 2 * n**2 * alpha2j - 2 * n**2
        numb1 = 4 * n * j - 2 * n * j * alpha2j + n * alpha2j
        numb0 = -(2 * j**2)
        denb1 = 2 * n**2 * alpha2j
        denb0 = -2 * n * j * alpha2j + n * alpha2j

        a, b = symbols("a, b")
        c = (2 * j - 1) / (2 * n)
        eq1 = Eq((numa2 * a**2 + numa1 * a + numa0) / (dena0 + dena1 * a), b)
        eq2 = Eq((numb2 * b**2 + numb1 * b + numb0) / (denb1 * b + denb0), a)

        try:
            nsol = nsolve([eq1, eq2], [a, b], [1, 1])

            if nsol[0] < (j - 1) / n and nsol[1] > j / n:
                return nsol[0], nsol[1], c
        except ValueError:
            pass

        try:
            sol = solve([eq1, eq2], [a, b])
        except ValueError:
            raise ValueError(
                f"Unsatisfiable alpha values {alphas}: could not solve for the"
                " triangular distribution parameters"
            )

        soln = [tuple(v.evalf() for v in s) for s in sol]

        for s in soln:
            s_a = s[0]
            s_b = s[1]

            if isinstance(s_a, Add):
                s_a = s_a.args[0]
            if isinstance(s_b, Add):
                s_b = s_b.args[0]

            if s_a < (j - 1) / n and s_b > j / n:
                return s_a, s_b, c

        raise ValueError(f"Could not find solution for {j=}, {alpha2j_1=}, {alpha2j=}")

    if verbose >= 3:
        print(f"{abc1(n,alphas[2])=}, {abcJ(n,alphas[2*n-1])=}")
        for i in range(2, n):
            print(f"{i=}  {abcj(n,i,alphas[2*i-1],alphas[2*i])}")

    params = []

    # Compute params for each interval (class)
    for j in range(1, n + 1):
        j_params = {}
        if j == 1:
            a, b, c = abc1(n, alphas[2])
            j_params["alpha2j_1"] = 0
            j_params["alpha2j"] = alphas[2]
        elif j == n:
            a, b, c = abcJ(n, alphas[2 * n - 1])
            j_params["alpha2j_1"] = alphas[2 * n - 1]
            j_params["alpha2j"] = 0
        else:
            a, b, c = abcj(n, j, alphas[2 * j - 1], alphas[2 * j])
            j_params["alpha2j_1"] = alphas[2 * j - 1]
            j_params["alpha2j"] = alphas[2 * j]

        j_params["a"] = a
        j_params["b"] = b
        j_params["c"] = c
        params.append(j_params)

    return params


def get_general_triangular_probabilities(n: int, alphas: np.ndarray, verbose: int = 0):
    """
    Get the probabilities for the general triangular distribution.

    Parameters
    ----------
    n : int
        Number of classes.
    alphas : np.ndarray
        Array of alphas.
    verbose : int, optional
        Verbosity level, by default 0.
    """

    intervals = get_intervals(n)
    probs = []

    # Compute probability for each interval (class) using the distribution function.
    params = get_general_triangular_params(n, alphas, verbose)

    for param in params:
        a = param["a"]
        b = param["b"]
        c = param["c"]
        j_probs = []

        for interval in intervals:
            j_probs.append(
                triangular_cdf(interval[1], a, b, c)
                - triangular_cdf(interval[0], a, b, c)
            )
            if verbose >= 2:
                print(f"\tinterval: {interval}, prob={j_probs[-1]}")

        probs.append(j_probs)

    return np.array(probs, dtype=np.float32)
