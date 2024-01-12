import numpy as np
from scipy.stats import binom


def get_binomial_probabilities(n):
    """Get probabilities from binominal distribution for n classes.

    Parameters
    ----------
    n : int
            Number of classes.

    Returns
    -------
    probs : 2d array-like
            Matrix of probabilities where each row represents the true class
            and each column the probability for class n.

    Example
    -------
    >>> get_binominal_probabilities(5)
    array([[6.561e-01, 2.916e-01, 4.860e-02, 3.600e-03, 1.000e-04],
            [2.401e-01, 4.116e-01, 2.646e-01, 7.560e-02, 8.100e-03],
            [6.250e-02, 2.500e-01, 3.750e-01, 2.500e-01, 6.250e-02],
            [8.100e-03, 7.560e-02, 2.646e-01, 4.116e-01, 2.401e-01],
            [1.000e-04, 3.600e-03, 4.860e-02, 2.916e-01, 6.561e-01]])
    """

    params = {}

    params["4"] = np.linspace(0.1, 0.9, 4)
    params["5"] = np.linspace(0.1, 0.9, 5)
    params["6"] = np.linspace(0.1, 0.9, 6)
    params["7"] = np.linspace(0.1, 0.9, 7)
    params["8"] = np.linspace(0.1, 0.9, 8)
    params["10"] = np.linspace(0.1, 0.9, 10)
    params["12"] = np.linspace(0.1, 0.9, 12)
    params["14"] = np.linspace(0.1, 0.9, 14)

    probs = []

    for true_class in range(0, n):
        probs.append(binom.pmf(np.arange(0, n), n - 1, params[str(n)][true_class]))

    return np.array(probs)
