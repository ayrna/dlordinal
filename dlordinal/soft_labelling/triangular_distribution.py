import math

import numpy as np

from .utils import get_intervals, triangular_cdf


def get_triangular_soft_labels(J: int, alpha2: float = 0.01, verbose: int = 0):
    """
    Get soft labels using triangular distributions for ``J`` classes or splits using
    the approach described in :footcite:t:`vargas2023softlabelling`.
    The :math:`[0,1]` interval is split into ``J`` intervals and the probability for
    each interval is computed as the difference between the value of the triangular
    distribution function for the interval boundaries. The probability for the first
    interval is computed as the value of the triangular distribution function for the
    first interval boundary.

    The triangular distribution function is denoted as :math:`\\text{p}(x, a, b, c)`
    where :math:`a`, :math:`b` and :math:`c` are the parameters of the distribution,
    and are determined by the number of classes :math:`J` and the value of the
    :math:`\\alpha_2` parameter. The value of :math:`\\alpha_2` represents the
    probability that is assigned to the adjacent classes of the target class. The
    parameters :math:`a`, :math:`b` and :math:`c` for class :math:`j` are computed
    as follows:

    .. math::
        a_j = \\begin{cases}
            0 & \\text{if } j = 1 \\\\
            \\frac{2j - 2 - 4j\\alpha_2 + 2\\alpha_2 \\pm
              \\sqrt{2\\alpha_2}}{2n(1 - 2\\alpha_2)} & \\text{if } 1 < j < J\\\\
            1 + \\frac{1}{\\pm J (\\sqrt{\\alpha_3} - 1)} & \\text{if } j = J
        \\end{cases}

    .. math::
        b_j = \\begin{cases}
            \\frac{1}{(1 - \\sqrt{\\alpha_1})n} & \\text{if } j = 1 \\\\
            \\frac{2j - 4j\\alpha_2 + 2\\alpha_2 \\pm
             \\sqrt{2\\alpha_2}}{2n(1 - 2\\alpha_2)} & \\text{if } 1 < j < J\\\\
             1 & \\text{if } j = J
        \\end{cases}

    .. math::
        c_j = \\begin{cases}
            0 & \\text{if } j = 1 \\\\
            \\frac{a + b}{2} & \\text{if } 1 < j  < J\\\\
            1 & \\text{if } j = J
        \\end{cases}

    The value of :math:`\\alpha_1`, that represents the error for the first class,
    is computed as follows:

    .. math::
        \\alpha_1 = \\left(\\frac{1 - \\sqrt{1 - 4(1 - 2\\alpha_2)(2\\alpha_2 -
          \\sqrt{2\\alpha_2})}}{2}\\right)^2

    The value of :math:`\\alpha_3`, that represents the error for the last class,
    is computed as follows:

    .. math::
        \\alpha_3 = \\left(\\frac{1 -
        \\sqrt{1 - 4\\left(\\frac{J - 1}{J}\\right)^2(1 - 2\\alpha_2)(\\sqrt{2\\alpha_2}
          (-1 + \\sqrt{2\\alpha_2}))}}{2}\\right)^2


    The value of :math:`\\alpha_2` is given by the user.

    Parameters
    ----------
    J : int
        Number of classes or splits (:math:`J`).
    alpha2 : float, optional, default=0.01
        Value of the :math:`\\alpha_2` parameter.
    verbose : int, optional, default=0
        Verbosity level.

    Raises
    ------
    ValueError
        If ``J`` is not a positive integer greater than 1.
        If ``alpha2`` is not a float between 0 and 1.

    Returns
    -------
    probs : 2d array-like of shape (J, J)
        Matrix of probabilities where each row represents the true class
        and each column the probability for class j.

    Example
    -------
    >>> from dlordinal.soft_labelling import get_triangular_soft_labels
    >>> get_triangular_soft_labels(5)
    array([[0.98845494, 0.01154505, 0.        , 0.        , 0.        ],
           [0.01      , 0.98      , 0.01      , 0.        , 0.        ],
           [0.        , 0.01      , 0.98      , 0.01      , 0.        ],
           [0.        , 0.        , 0.01      , 0.98      , 0.01      ],
           [0.        , 0.        , 0.        , 0.00505524, 0.99494475]])
    """

    if J < 2 or not isinstance(J, int):
        raise ValueError(f"{J=} must be a positive integer greater than 1")

    if alpha2 < 0 or alpha2 > 1:
        raise ValueError(f"{alpha2=} must be a float between 0 and 1")

    if verbose >= 1:
        print(f"Computing triangular probabilities for {J=} and {alpha2=}...")

    def compute_alpha1(alpha2):
        c_minus = (1 - 2 * alpha2) * (2 * alpha2 - math.sqrt(2 * alpha2))

        return pow((1 - math.sqrt(1 - 4 * c_minus)) / 2, 2)

    def compute_alpha3(alpha2):
        c1 = (
            pow((J - 1) / J, 2)
            * (1 - 2 * alpha2)
            * (math.sqrt(2 * alpha2) * (-1 + math.sqrt(2 * alpha2)))
        )

        return pow((1 - math.sqrt(1 - 4 * c1)) / 2, 2)

    alpha1 = compute_alpha1(alpha2)
    alpha3 = compute_alpha3(alpha2)

    if verbose >= 1:
        print(f"{alpha1=}, {alpha2=}, {alpha3=}")

    def b1(J):
        return 1.0 / ((1.0 - math.sqrt(alpha1)) * J)

    def aj(J, j):
        num1 = 2.0 * j - 2 - 4 * j * alpha2 + 2 * alpha2
        num2 = math.sqrt(2 * alpha2)
        den = 2.0 * J * (1 - 2 * alpha2)

        max_value = (j - 1.0) / J

        # +-
        return (
            (num1 + num2) / den
            if (num1 + num2) / den < max_value
            else (num1 - num2) / den
        )

    def bj(J, j):
        num1 = 2.0 * j - 4 * j * alpha2 + 2 * alpha2
        num2 = math.sqrt(2 * alpha2)
        den = 2.0 * J * (1 - 2 * alpha2)

        min_value = j / J

        # +-
        return (
            (num1 + num2) / den
            if (num1 + num2) / den > min_value
            else (num1 - num2) / den
        )

    def aJ(J):
        aJ_plus = 1.0 + 1.0 / (J * (math.sqrt(alpha3) - 1.0))
        aJ_minus = 1.0 + 1.0 / (-J * (math.sqrt(alpha3) - 1.0))
        return aJ_plus if aJ_plus > 0.0 else aJ_minus

    if verbose >= 3:
        print(f"{b1(J)=}, {aJ(J)=}, {aj(J, 1)=}, {bj(J,1)=}")
        for i in range(1, J + 1):
            print(f"{i=}  {aj(J, i)=}, {bj(J,i)=}")

    intervals = get_intervals(J)
    probs = []

    # Compute probability for each interval (class) using the distribution function.
    for j in range(1, J + 1):
        j_probs = []
        if j == 1:
            a = 0.0
            b = b1(J)
            c = 0.0
        elif j == J:
            a = aJ(J)
            b = 1.0
            c = 1.0
        else:
            a = aj(J, j)
            b = bj(J, j)
            c = (a + b) / 2.0

        if verbose >= 1:
            print(f"Class: {j}, {a=}, {b=}, {c=}, (j-1)/J={(j-1)/J}, (j/J)={j/J}")

        for interval in intervals:
            j_probs.append(
                triangular_cdf(interval[1], a, b, c)
                - triangular_cdf(interval[0], a, b, c)
            )
            if verbose >= 2:
                print(f"\tinterval: {interval}, prob={j_probs[-1]}")

        probs.append(j_probs)

    return np.array(probs)
