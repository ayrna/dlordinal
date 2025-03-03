from typing import Union

import numpy as np


def get_geometric_soft_labels(J: int, alphas: Union[float, list] = 0.1):
    """
    Get soft labels based on the discrete geometric distribution according to :footcite:t:`haas2023geometric`.


    Parameters
    ----------
    J : int
        Number of classes.
    alphas : float or list, default=0.1
        The smoothing factor(s) for geometric distribution-based unimodal smoothing.

        - **Single alpha value**:
          When a single alpha value in the range `[0, 1]`, e.g., `0.1`, is provided,
          all classes will be smoothed equally and symmetrically.
          This is done by deducting alpha from the actual class, :math:`1 - \\alpha`,
          and allocating :math:`\\alpha` to the rest of the classes, decreasing monotonically
          from the actual class in the form of the geometric distribution.

          Formula ( with :math:`j` as the index of the observed class in the one-hot encoded label and :math:`k` the current class):

          .. math::
            p_{i}^G(k) = \\begin{cases}
                1-\\alpha  & \\text{if } k = j \\\\
                1/G_{i} \\;  \\alpha^{|j-k|+1}(1-\\alpha) & \\text{if } k \\neq j  \\\\
            \\end{cases}.

          Normalizing constant:

          .. math::
            G_{i} = p_{i}^G(k \\neq j)  =  \\sum_{k \\neq j} \\alpha^{|j-k|}(1-\\alpha).

        - **List of alpha values**:
          Alternatively, a list of size :attr:`num_classes` can be provided to specify class-wise symmetric
          smoothing factors. An example for five classes is: `[0.2, 0.05, 0.1, 0.15, 0.1]`.

        - **List of smoothing relations**:
          To control the fraction of the left-over probability mass :math:`\\alpha` allocated to the left
          (:math:`F_l \\in [0,1]`) and right (:math:`F_r \\in [0,1]`) sides of the true class, with
          :math:`F_l + F_r = 1`, a list of smoothing relations of the form :math:`(\\alpha, F_l, F_r)`
          can be specified. This enables asymmetric unimodal smoothing. An example for five classes is:
          `[(0.2, 0.0, 1.0), (0.05, 0.8, 0.2), (0.1, 0.5, 0.5), (0.15, 0.6, 0.4), (0.1, 1.0, 0.0)]`.

          .. math::
            p_{i}^G(k) = \\begin{cases}
                1-\\alpha_{j}  & \\text{if } k = j \\\\
                1/G_{i} \\; F_{l,j} \\;  \\alpha_{j}^{(j-k)+1}(1-\\alpha_{j}) & \\text{if } k < j \\\\
                1/G_{i} \\; F_{r,j} \\;  \\alpha_{j}^{(k-j)+1}(1-\\alpha_{j}) & \\text{if } k > j  \\\\
            \\end{cases}


    Raises
    ------
    ValueError
        If ``J`` is not a positive integer greater than 1.
        If smoothing values in ``alphas`` are not in [0,1].
        If ``alphas`` is a list and size of ``alphas`` is not equal to ``J``.
        If ``alphas`` is not a float, list of floats, or list of tuples.
        If probability fractions :math:`F_l \\in [0,1]` and :math:`F_r \\in [0,1]` do not sum to one.

    Returns
    -------
    probs : 2d array-like of shape (J, J)
        Matrix of probabilities where each row represents the true class
        and each column the probability for class j.

    Example
    -------
    >>> from dlordinal.soft_labelling import get_geometric_soft_labels
    >>> get_geometric_soft_labels(5)
    array([[0.9       , 0.090009  , 0.0090009 , 0.00090009, 0.00009001],
       [0.04739336, 0.9       , 0.04739336, 0.00473934, 0.00047393],
       [0.00454545, 0.04545455, 0.9       , 0.04545455, 0.00454545],
       [0.00047393, 0.00473934, 0.04739336, 0.9       , 0.04739336],
       [0.00009001, 0.00090009, 0.0090009 , 0.090009  , 0.9       ]])
    >>> get_geometric_soft_labels(5, alphas=0.3)
    array([[0.7       , 0.21171489, 0.06351447, 0.01905434, 0.0057163 ],
       [0.12552301, 0.7       , 0.12552301, 0.0376569 , 0.01129707],
       [0.03461538, 0.11538462, 0.7       , 0.11538462, 0.03461538],
       [0.01129707, 0.0376569 , 0.12552301, 0.7       , 0.12552301],
       [0.0057163 , 0.01905434, 0.06351447, 0.21171489, 0.7       ]])
    >>> get_geometric_soft_labels(5, alphas=[0.3,0.2,0.05,0.02,0.5])
    array([[0.7       , 0.21171489, 0.06351447, 0.01905434, 0.0057163 ],
       [0.08928571, 0.8       , 0.08928571, 0.01785714, 0.00357143],
       [0.00119048, 0.02380952, 0.95      , 0.02380952, 0.00119048],
       [0.00000396, 0.00019798, 0.00989903, 0.98      , 0.00989903],
       [0.03333333, 0.06666667, 0.13333333, 0.26666667, 0.5       ]])
    >>> get_geometric_soft_labels(5, alphas=[(0.2, 0.0, 1.0), (0.05, 0.8, 0.2), (0.1, 0.5, 0.5), (0.15, 0.6, 0.4), (0.1, 1.0, 0.0)])
    array([[0.8       , 0.16025641, 0.03205128, 0.00641026, 0.00128205],
       [0.04      , 0.95      , 0.00950119, 0.00047506, 0.00002375],
       [0.00454545, 0.04545455, 0.9       , 0.04545455, 0.00454545],
       [0.00172708, 0.01151386, 0.07675906, 0.85      , 0.06      ],
       [0.00009001, 0.00090009, 0.0090009 , 0.090009  , 0.9       ]])
    """

    if not isinstance(J, int) or J < 2:
        raise ValueError(f"J={J} must be a positive integer greater than 1")

    if isinstance(alphas, list) and len(alphas) != J:
        raise ValueError(f"Size of alphas={len(alphas)} must be equal to J={J}")

    if (
        not isinstance(alphas, float)
        and not (
            isinstance(alphas, list) and all(isinstance(item, float) for item in alphas)
        )
        and not (
            isinstance(alphas, list)
            and all(isinstance(item, tuple) for item in alphas)
            and all(len(item) == 3 for item in alphas)
        )
    ):
        raise ValueError(
            f"alphas={alphas} must either be a single float value [0,1],"
            " a list of floats, or tuples of the form (alpha,F_l,F_r)"
        )

    if (
        (isinstance(alphas, float) and (alphas < 0 or alphas > 1))
        or (
            isinstance(alphas, list)
            and all(isinstance(item, float) for item in alphas)
            and any((alpha < 0 or alpha > 1) for alpha in alphas)
        )
        or (
            isinstance(alphas, list)
            and all(isinstance(item, tuple) for item in alphas)
            and any((alpha[0] < 0 or alpha[0] > 1) for alpha in alphas)
        )
    ):
        raise ValueError(f"alphas={alphas} must be in the range [0, 1]")

    if (
        isinstance(alphas, list)
        and all(isinstance(item, tuple) for item in alphas)
        and any((alpha[1] + alpha[2] != 1.0) for alpha in alphas)
    ):
        raise ValueError(f"F_l and F_r must sum to one, alphas={alphas}")

    if isinstance(alphas, list) and all(isinstance(item, tuple) for item in alphas):
        probs = _get_asymmetric_geometric_soft_labels(J, alphas)
    else:
        probs = _get_symmetric_geometric_soft_labels(J, alphas)

    return probs


def _get_asymmetric_geometric_soft_labels(J, alphas):
    probs = np.zeros((J, J))
    for y in range(J):
        # If alpha = 0.0 --> one-hot encoding
        if alphas[y][0] == 0.0:
            probs[y, :] = np.array([1 if k == y else 0 for k in range(J)])
            continue

        # Normalizing constants
        G_left = sum(
            [(pow(alphas[y][0], abs(y - i)) * (1 - alphas[y][0])) for i in range(0, y)]
        )
        G_right = sum(
            [
                (pow(alphas[y][0], abs(y - i)) * (1 - alphas[y][0]))
                for i in range(y + 1, J)
            ]
        )

        # Fraction of alpha supposed to go left and right of y
        # incl. fix for edge cases for first and last class
        if y == (J - 1):
            F_left = 1
            F_right = 0
        elif y == 0:
            F_left = 0
            F_right = 1
        else:
            F_left = alphas[y][1]
            F_right = alphas[y][2]

        for k in range(J):
            if k == y:
                probs[y, k] = 1 - alphas[y][0]
                # Fix edge cases for first and last class
                if F_left == 0 and k == (J - 1):
                    probs[y, k] = 1.0
                if F_right == 0 and k == 0:
                    probs[y, k] = 1.0
            elif k < y:  # left
                probs[y, k] = (
                    F_left
                    * alphas[y][0]
                    * (1 / G_left * pow(alphas[y][0], (y - k)) * (1 - alphas[y][0]))
                    if F_left > 0
                    else 0
                )
            elif k > y:  # right
                probs[y, k] = (
                    F_right
                    * alphas[y][0]
                    * (1 / G_right * pow(alphas[y][0], (k - y)) * (1 - alphas[y][0]))
                    if F_right > 0
                    else 0
                )
    return probs


def _get_symmetric_geometric_soft_labels(J, alphas):
    probs = np.zeros((J, J))
    for y in range(J):
        # Determine smoothing factor alpha for true class y
        if isinstance(alphas, list):
            alpha = alphas[y]
        else:
            alpha = alphas

        # If alpha = 0.0 --> one-hot encoding
        if alpha == 0.0:
            probs[y, :] = np.array([1 if k == y else 0 for k in range(J)])
            continue

        # Calculate normalizing constant
        G = sum(
            [
                (pow(alpha, abs(y - k)) * (1 - alpha)) if k != y else 0
                for k in range(0, J)
            ]
        )

        # Set soft labels for class y
        for k in range(J):
            if y == k:  # true class
                probs[y, k] = 1 - alpha
            else:  # other classes
                probs[y, k] = alpha * (1 / G * pow(alpha, (abs(y - k))) * (1 - alpha))
    return probs
