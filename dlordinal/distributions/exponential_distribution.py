import numpy as np
from scipy.special import softmax


def get_exponential_probabilities(n, p=1.0, tau=1.0):
    """Get probabilities from exponential distribution for n classes.

    Parameters
    ----------
    n : int
            Number of classes.
    p : float, default=1.0
            Exponent parameter.
    tau : float, default=1.0
            Scaling parameter.

    Returns
    -------
    probs : 2d array-like
            Matrix of probabilities where each row represents the true class
            and each column the probability for class n.

    Example
    -------
    >>> get_exponential_probabilities(5)
    array([[0.63640865, 0.23412166, 0.08612854, 0.03168492, 0.01165623],
    [0.19151597, 0.52059439, 0.19151597, 0.07045479, 0.02591887],
    [0.06745081, 0.1833503 , 0.49839779, 0.1833503 , 0.06745081],
    [0.02591887, 0.07045479, 0.19151597, 0.52059439, 0.19151597],
    [0.01165623, 0.03168492, 0.08612854, 0.23412166, 0.63640865]])
    """

    probs = []

    for true_class in range(0, n):
        probs.append(-(np.abs(np.arange(0, n) - true_class) ** p) / tau)

    return softmax(np.array(probs), axis=1)
