import math

import numpy as np
from sympy import Eq, nsolve, solve, symbols
from sympy.core.add import Add

from .utils import get_intervals, triangular_cdf


def get_general_triangular_params(J: int, alphas: np.ndarray, verbose: int = 0):
    """
    Get the parameters (:math:`a`, :math:`b` and :math:`c`) for the
    general triangular distribution. The parameters are computed using
    the :math:`\\boldsymbol{\\alpha}` values and the number of classes
    :math:`J`. The :math:`\\boldsymbol{\\alpha}` vector contains two :math:`\\alpha`
    values for each class or split. :math:`\\alpha_0` should always be 0, given that
    the error on the left side of the first class is always 0. In the same way, the
    :math:`\\alpha_{2J}` value should always be 0, given that the error on the right
    side of the last class is always 0. The :math:`\\alpha` values for the other
    classes should be between 0 and 1.

    The parameters :math:`a`, :math:`b` and :math:`c` for class :math:`j` are computed
    as described in :footcite:t:`vargas2023gentri`.

    Parameters
    ----------
    J : int
        Number of classes or splits.
    alphas : np.ndarray
        Array that represents the error on the left and right side of each class.
        It is the :math:`\\boldsymbol{\\alpha}` vector described
        in :footcite:t:`vargas2023gentri`.
    verbose : int, optional
        Verbosity level, by default 0.

    Raises
    ------
    ValueError
        If the number of classes :math:`J` is less than 2.
        If the :math:`\\boldsymbol{\\alpha}` vector is not a numpy array of shape
        :math:`(2J,)`.

    Returns
    -------
    list
        List of dictionaries with the parameters for each class.

    Example
    -------
    >>> from dlordinal.soft_labelling import get_general_triangular_params
    >>> get_general_triangular_params(5, [0, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0])
    [{'alpha2j_1': 0, 'alpha2j': 0.05, 'a': 0, 'b': 0.25760143110525874, 'c': 0}, {'alpha2j_1': 0.05, 'alpha2j': 0.05, 'a': 0.153752470442574, 'b': 0.446247529557426, 'c': 0.3}, {'alpha2j_1': 0.05, 'alpha2j': 0.05, 'a': 0.353752470442574, 'b': 0.646247529557426, 'c': 0.5}, {'alpha2j_1': 0.05, 'alpha2j': 0.1, 'a': 0.550779686438060, 'b': 0.875486049708105, 'c': 0.7}, {'alpha2j_1': 0.0, 'alpha2j': 0, 'a': 0.8, 'b': 1, 'c': 1}]

    """

    alphas = np.array(alphas)
    if not isinstance(alphas, np.ndarray) or alphas.shape != (2 * J,):
        raise ValueError(
            f"alphas must be a numpy array of shape (2 * n,), but got {alphas.shape}"
        )

    if J < 2:
        raise ValueError(f"J must be greater than 1, but got {J}")

    def abc1(J, alpha):
        a = 0
        b = 1.0 / ((1.0 - math.sqrt(alpha)) * J)
        c = 0
        return a, b, c

    def abcJ(J, alpha):
        a = 1.0 - 1.0 / (J * (1.0 - math.sqrt(alpha)))
        b = 1
        c = 1
        return a, b, c

    def abcj(J, j, alpha2j_1, alpha2j):
        # Calculation of the terms of the system of equations
        # b = ...
        numa2 = 2 * J**2 - 2 * J**2 * alpha2j_1
        numa1 = 2 * j * J * alpha2j_1 - J * alpha2j_1 - 4 * J * j + 4 * J
        numa0 = 2 * j**2 - 4 * j + 2
        dena0 = 2 * j * J * alpha2j_1 - J * alpha2j_1
        dena1 = -(2 * J**2 * alpha2j_1)

        # a = ...
        numb2 = 2 * J**2 * alpha2j - 2 * J**2
        numb1 = 4 * J * j - 2 * J * j * alpha2j + J * alpha2j
        numb0 = -(2 * j**2)
        denb1 = 2 * J**2 * alpha2j
        denb0 = -2 * J * j * alpha2j + J * alpha2j

        a, b = symbols("a, b")
        c = (2 * j - 1) / (2 * J)
        eq1 = Eq((numa2 * a**2 + numa1 * a + numa0) / (dena0 + dena1 * a), b)
        eq2 = Eq((numb2 * b**2 + numb1 * b + numb0) / (denb1 * b + denb0), a)

        try:
            nsol = nsolve([eq1, eq2], [a, b], [1, 1])

            if nsol[0] < (j - 1) / J and nsol[1] > j / J:
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

            if s_a < (j - 1) / J and s_b > j / J:
                return s_a, s_b, c

        raise ValueError(f"Could not find solution for {j=}, {alpha2j_1=}, {alpha2j=}")

    if verbose >= 3:
        print(f"{abc1(J,alphas[2])=}, {abcJ(J,alphas[2*J-1])=}")
        for i in range(2, J):
            print(f"{i=}  {abcj(J,i,alphas[2*i-1],alphas[2*i])}")

    params = []

    # Compute params for each interval (class)
    for j in range(1, J + 1):
        j_params = {}
        if j == 1:
            a, b, c = abc1(J, alphas[2])
            j_params["alpha2j_1"] = 0
            j_params["alpha2j"] = alphas[2]
        elif j == J:
            a, b, c = abcJ(J, alphas[2 * J - 1])
            j_params["alpha2j_1"] = alphas[2 * J - 1]
            j_params["alpha2j"] = 0
        else:
            a, b, c = abcj(J, j, alphas[2 * j - 1], alphas[2 * j])
            j_params["alpha2j_1"] = alphas[2 * j - 1]
            j_params["alpha2j"] = alphas[2 * j]

        j_params["a"] = a
        j_params["b"] = b
        j_params["c"] = c
        params.append(j_params)

    return params


def get_general_triangular_soft_labels(J: int, alphas: np.ndarray, verbose: int = 0):
    """Get soft labels using triangular distributions for ``J`` classes or splits.
    The :math:`[0,1]` interval is split into ``J`` intervals and the probability for
    each interval is computed as the difference between the value of the triangular
    distribution function for the interval boundaries. The probability for the first
    interval is computed as the value of the triangular distribution function for the
    first interval boundary.

    The triangular distribution function is denoted as :math:`\\text{f}(x, a, b, c)`.
    The parameters :math:`a`, :math:`b` and :math:`c` for class :math:`j` are computed
    as described in :footcite:t:`vargas2023gentri`.

    Parameters
    ----------
    J : int
        Number of classes.
    alphas : np.ndarray
        Array of alphas.
    verbose : int, optional
        Verbosity level, by default 0.

    Raises
    ------
    ValueError
        If ``J`` is not a positive integer greater than 1.
        If ``alphas`` is not a numpy array of shape :math:`(2J,)`.

    Returns
    -------
    probs : 2d array-like of shape (J, J)
        Matrix of probabilities where each row represents the true class
        and each column the probability for class j.

    Example
    -------
    >>> from dlordinal.soft_labelling import get_general_triangular_soft_labels
    >>> get_general_triangular_soft_labels(
    ...     5,
    ...     [0, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0]
    ... )
    array([[0.95, 0.05, 0.  , 0.  , 0.  ],
           [0.05, 0.9 , 0.05, 0.  , 0.  ],
           [0.  , 0.05, 0.9 , 0.05, 0.  ],
           [0.  , 0.  , 0.05, 0.85, 0.1 ],
           [0.  , 0.  , 0.  , 0.  , 1.  ]], dtype=float32)
    """

    if J < 2:
        raise ValueError(f"J must be greater than 1, but got {J}")

    if isinstance(alphas, list):
        alphas = np.array(alphas)

    if not isinstance(alphas, np.ndarray) or alphas.shape != (2 * J,):
        raise ValueError(
            "alphas must be a numpy array or list of shape (2 * n,),"
            " but got {alphas.shape}"
        )

    intervals = get_intervals(J)
    probs = []

    # Compute probability for each interval (class) using the distribution function.
    params = get_general_triangular_params(J, alphas, verbose)

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
