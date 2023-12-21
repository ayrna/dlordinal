import numpy as np
from scipy.special import softmax
from scipy.stats import poisson


def get_poisson_probabilities(n):
    """Get probabilities from poisson distribution for n classes.

    Parameters
    ----------
    n : int
        Number of classes.

    Returns
    -------
    probs : 2d array-like
        Matrix of probabilities where each row represents the true class
        and each column the probability for class n.
    """

    probs = []

    for true_class in range(1, n + 1):
        probs.append(poisson.pmf(np.arange(0, n), true_class))

    return softmax(np.array(probs), axis=1)
