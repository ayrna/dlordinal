import json
import os
from math import sqrt
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import recall_score


def _to_numpy(x: ArrayLike, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Helper function to safely convert input to NumPy array (supports torch tensors on CPU/CUDA).
    Parameters
    ----------
    x : numpy array-like
        Input data to be converted to a NumPy array.
    dtype : numpy data-type, optional
        Desired data-type for the array. If not provided, the data type of the input
        is used.

    Returns
    -------
    np.ndarray
        The input converted to a NumPy array.

    Notes
    -----
    This function can be extended to support other types as needed.

    Examples
    --------
    >>> import torch
    >>> from dlordinal.metrics.metrics import _to_numpy
    >>> import numpy as np
    >>> # tensor on GPU that requires gradients
    >>> tensor_gpu = torch.tensor([1., 2., 3.], device='cuda', requires_grad=True)
    >>> try:
    ...     np.array(tensor_gpu)
    ... except Exception as e:
    ...     print(f"Error converting tensor on GPU to NumPy directly: {e}")
    >>> # using the _to_numpy function
    >>> _to_numpy(tensor_gpu)
    array([1., 2., 3.], dtype=float32)

    """
    try:
        import torch

        is_torch = isinstance(x, torch.Tensor)
    except ImportError:
        is_torch = False

    if is_torch:
        # detach(): removes tensor from the computation graph (no gradients tracked)
        # cpu(): moves the tensor to CPU memory if it’s on GPU
        # Convert to a NumPy array to avoid forwarding torch.Tensor.__array__ into
        # NumPy which may trigger the deprecation about the `copy` keyword in
        # NumPy 2.0. Using `.numpy()` produces a plain ndarray we can safely wrap
        # with np.asarray below.
        x = x.detach().cpu().numpy()

    # Use np.asarray instead of np.array to avoid calling an object's
    # `__array__` with unexpected keywords (NumPy 2.0 migration warning).
    if dtype is not None:
        return np.asarray(x, dtype=dtype)
    return np.asarray(x)


def _to_label_array(y: ArrayLike) -> np.ndarray:
    """Converts labels, one-hot targets or predicted probabilities to a 1D array
    of labels.

    Two-dimensional inputs are interpreted as one-hot targets or probabilities and
    are reduced with ``argmax`` *before* any type conversion, so that probabilities
    are never truncated.

    Parameters
    ----------
    y : array-like
        Labels of shape (N,), or one-hot targets / probabilities of shape (N, J).

    Returns
    -------
    np.ndarray
        Array of labels of shape (N,).
    """
    y = _to_numpy(y)
    if y.ndim > 1:
        y = np.argmax(y, axis=1)
    return y


def _per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Computes the sensitivity (recall) of every class present in ``y_true``.

    Predicted labels that never occur in ``y_true`` have an undefined sensitivity and
    must not be reported as a sensitivity of zero, so only the classes in ``y_true``
    are considered. The returned array is ordered by increasing label.
    """
    return np.asarray(
        recall_score(y_true, y_pred, labels=np.unique(y_true), average=None)
    )


def ranked_probability_score(y_true, y_proba):
    """Computes the ranked probability score as presented in :footcite:t:`janitza2016random`.

    Parameters
    ----------
    y_true : array-like
            Target labels.
    y_proba : array-like
            Predicted probabilities.

    Returns
    -------
    rps : float
            The ranked probability score.

    Examples
    --------
    >>> import numpy as np
    >>> from dlordinal.metrics import ranked_probability_score
    >>> y_true = np.array([0, 0, 3, 2])
    >>> y_pred = np.array([[0.2, 0.4, 0.2, 0.2], [0.7, 0.1, 0.1, 0.1], [0.5, 0.05, 0.1, 0.35], [0.1, 0.05, 0.65, 0.2]])
    >>> ranked_probability_score(y_true, y_pred)
    0.5068750000000001
    """
    y_true = _to_label_array(y_true).astype(int)
    y_proba = _to_numpy(y_proba, dtype=float)

    n_classes = y_proba.shape[1]
    if y_true.size and (y_true.min() < 0 or y_true.max() >= n_classes):
        raise ValueError(
            f"y_true must contain labels in [0, {n_classes - 1}], the range of"
            " classes covered by y_proba."
        )

    y_oh = np.zeros(y_proba.shape)
    y_oh[np.arange(len(y_true)), y_true] = 1

    # Vectorised over samples: squared distance between the cumulative distributions
    cdf_diff = y_proba.cumsum(axis=1) - y_oh.cumsum(axis=1)
    return float(np.mean(np.sum(cdf_diff**2, axis=1)))


def minimum_sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the sensitivity by class and returns the lowest value.

    Parameters
    ----------
    y_true : array-like
            Target labels.
    y_pred : array-like
            Predicted probabilities or labels.

    Returns
    -------
    ms : float
            Minimum sensitivity.

    Examples
    --------
    >>> import numpy as np
    >>> from dlordinal.metrics import minimum_sensitivity
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> minimum_sensitivity(y_true, y_pred)
    0.5
    """
    y_true = _to_label_array(y_true)
    y_pred = _to_label_array(y_pred)

    sensitivities = _per_class_recall(y_true, y_pred)
    return np.min(sensitivities)


def accuracy_off1(y_true: np.ndarray, y_pred: np.ndarray, labels=None) -> float:
    """Computes the accuracy of the predictions, allowing errors if they occur in an
    adjacent class.

    Parameters
    ----------
    y_true : array-like
            Target labels.
    y_pred : array-like
            Predicted probabilities or labels.
    labels : array-like or None
            Ordered list of all the classes. Two classes are adjacent when they are
            consecutive in this list. Every label in ``y_true`` and ``y_pred`` must be
            included. If None, the labels are taken as numeric values and two classes
            are adjacent when their labels differ by at most one, which is the usual
            case of integer labels :math:`0, \\ldots, J-1`.

    Returns
    -------
    acc : float
            1-off accuracy.

    Raises
    ------
    ValueError
            If ``labels`` is given and does not include every label of ``y_true`` and
            ``y_pred``.

    Examples
    --------
    >>> import numpy as np
    >>> from dlordinal.metrics import accuracy_off1
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 0, 0, 1])
    >>> accuracy_off1(y_true, y_pred)
    0.8571428571428571
    """
    y_true = _to_label_array(y_true)
    y_pred = _to_label_array(y_pred)

    if labels is None:
        # Distances are measured on the label values, so classes that are absent
        # from the data cannot make two distant classes look adjacent.
        return float(np.mean(np.abs(y_true - y_pred) <= 1))

    labels = _to_numpy(labels)
    sorter = np.argsort(labels)

    def _positions(y):
        idx = np.clip(np.searchsorted(labels, y, sorter=sorter), 0, len(labels) - 1)
        pos = sorter[idx]
        if not np.all(labels[pos] == y):
            raise ValueError(
                "labels must include every label of y_true and y_pred, missing:"
                f" {np.setdiff1d(y, labels)}"
            )
        return pos

    return float(np.mean(np.abs(_positions(y_true) - _positions(y_pred)) <= 1))


def gmsec(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Geometric Mean of the Sensitivity of the Extreme Classes (GMSEC). It was proposed
    in (:footcite:t:`vargas2024improving`) with the aim of assessing the performance of
    the classification performance for the first and the last classes.

    Parameters
    ----------
    y_true : array-like
            Target labels.
    y_pred : array-like
            Predicted probabilities or labels.

    Returns
    -------
    gmec : float
            Geometric mean of the sensitivities of the extreme classes.

    Examples
    --------
    >>> import numpy as np
    >>> from dlordinal.metrics import gmsec
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> gmsec(y_true, y_pred)
    0.7071067811865476
    """
    y_true = _to_label_array(y_true)
    y_pred = _to_label_array(y_pred)

    sensitivities = _per_class_recall(y_true, y_pred)
    return np.sqrt(sensitivities[0] * sensitivities[-1])


def amae(y_true: np.ndarray, y_pred: np.ndarray):
    """Computes the average mean absolute error computed independently for each class
    as presented in :footcite:t:`baccianella2009evaluation`.

    Parameters
    ----------
    y_true : array-like
            Targets labels with one-hot or integer encoding.
    y_pred : array-like
            Predicted probabilities or labels.

    Returns
    -------
    amae : float
            Average mean absolute error.

    Examples
    --------
    >>> import numpy as np
    >>> from dlordinal.metrics import amae
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> amae(y_true, y_pred)
    0.125
    """
    return np.mean(_per_class_mae(y_true, y_pred))


def mmae(y_true: np.ndarray, y_pred: np.ndarray):
    """Computes the maximum mean absolute error computed independently for each class
    as presented in :footcite:t:`cruz2014metrics`.

    Parameters
    ----------
    y_true : array-like
            Target labels with one-hot or integer encoding.
    y_pred : array-like
            Predicted probabilities or labels.

    Returns
    -------
    mmae : float
            Maximum mean absolute error.

    Examples
    --------
    >>> import numpy as np
    >>> from dlordinal.metrics import mmae
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> mmae(y_true, y_pred)
    0.5
    """
    return _per_class_mae(y_true, y_pred).max()


def _per_class_mae(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """Computes the mean absolute error of every class present in ``y_true``.

    The absolute error is measured on the label values, so classes that do not appear
    in ``y_true`` or ``y_pred`` do not shrink the distance between the classes that do.
    The returned array is ordered by increasing label.
    """
    y_true = _to_label_array(y_true)
    y_pred = _to_label_array(y_pred)

    classes, inverse = np.unique(y_true, return_inverse=True)
    abs_errors = np.abs(y_true - y_pred).astype(np.float64)
    error_sums = np.bincount(inverse, weights=abs_errors, minlength=len(classes))
    counts = np.bincount(inverse, minlength=len(classes))
    return error_sums / counts


def mes(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the Mean of the Sensitivities of the first and last class (MES).
    This metric is useful for assessing the balanced performance between the extreme classes
    using arithmetic mean.

    Parameters
    ----------
    y_true : array-like
            Target labels.
    y_pred : array-like
            Predicted probabilities or labels.

    Returns
    -------
    mes : float
            Mean of the sensitivities of the extreme classes.

    Examples
    --------
    >>> import numpy as np
    >>> from dlordinal.metrics import mes
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> mes(y_true, y_pred)
    0.75
    """
    # argmax is applied before the integer cast, so probabilities are not truncated
    y_true = _to_label_array(y_true).astype(int)
    y_pred = _to_label_array(y_pred).astype(int)

    sensitivities = _per_class_recall(y_true, y_pred)
    mes = (sensitivities[0] + sensitivities[-1]) / 2.0
    return mes


def gmes(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the Geometric Mean of the Sensitivities of the first and last class (GMES).
    This metric is useful for assessing the balanced performance between the extreme classes
    using geometric mean.

    Parameters
    ----------
    y_true : array-like
            Target labels.
    y_pred : array-like
            Predicted probabilities or labels.

    Returns
    -------
    gmes : float
            Geometric mean of the sensitivities of the extreme classes.

    Examples
    --------
    >>> import numpy as np
    >>> from dlordinal.metrics import gmes
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> gmes(y_true, y_pred)
    0.7071067811865476
    """
    # argmax is applied before the integer cast, so probabilities are not truncated
    y_true = _to_label_array(y_true).astype(int)
    y_pred = _to_label_array(y_pred).astype(int)

    sensitivities = _per_class_recall(y_true, y_pred)
    gmes = sqrt(sensitivities[0] * sensitivities[-1])
    return gmes


def write_metrics_dict_to_file(
    metrics: Dict[str, float],
    path_str: str,
    filter_fn: Optional[Callable[[str, float], bool]] = lambda n, v: True,
) -> None:
    """Writes a dictionary of metrics to a tabular file.
    The dictionary is filtered by the filter function.
    The first time that the metrics are saved to the file, the keys are written as
    the header. Subsequent calls append the values to the file.

    Parameters
    ----------
    metrics : Dict[str, float]
            Dictionary of metric names associated with their value.
    path_str : str
            Path to the file that will be saved.
            The directory of the file will be created if it does not exist.
            If the file exists, the metrics will be appended to the file in a new row.
    filter_fn : Optional[Callable[[str, bool], bool]], default=lambda n, v: True
            Function that filters the metrics.
            The function takes the name and the value of the metric and returns ``True``
            if the metric should be saved.

    Examples
    --------
    >>> metrics = {'acc': 0.5, 'gmsec': 0.25}
    >>> write_metrics_dict_to_file(metrics, 'results.txt')
    >>> write_metrics_dict_to_file(metrics, 'results.txt')
    >>> with open('results.txt', 'r') as f:
    ...     print(f.read())
    acc	gmsec
    0.5	0.25
    0.5	0.25

    >>> write_metrics_dict_to_file(metrics, 'results.txt', filter_fn=lambda name, value: name == 'acc')
    >>> with open('results.txt', 'r') as f:
    ...     print(f.read())
    acc
    0.5
    0.5
    """

    path = Path(path_str)
    directory = path.parents[0]
    os.makedirs(directory, exist_ok=True)

    if not path.is_file():
        with open(path, "w") as f:
            for k, v in metrics.items():
                if filter_fn(k, v):
                    f.write(f"{k},")
            f.write("\n")

    with open(path, "a") as f:
        for k, v in metrics.items():
            if filter_fn(k, v):
                f.write(f"{v},")
        f.write("\n")


def write_array_to_file(array: np.ndarray, path_str: str, id: str):
    """Writes an array to a json file.
    The array is saved as a dictionary with the key 'id' and the value 'array'.

    Parameters
    ----------
    array : array-like
            Array to be saved.
    path_str : str
            Path to the file that will be saved.
            The directory of the file will be created if it does not exist.
    id : str
            Id of the array.

    Examples
    --------
    >>> array = np.array([0, 1, 2])
    >>> write_array_to_file(array, 'results.json', 'array')
    >>> with open('results.json', 'r') as f:
    ...     print(f.read())
    {"array": [0, 1, 2]}

    >>> array2 = np.array([3, 4, 5])
    >>> write_array_to_file(array, 'results.json', 'array2')
    >>> with open('results.json', 'r') as f:
    ...     print(f.read())
    {"array": [0, 1, 2], "array2": [3, 4, 5]}
    """

    path = Path(path_str)
    directory = path.parents[0]
    os.makedirs(directory, exist_ok=True)

    if path.is_file():
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = dict()

    data[id] = array.tolist()

    with open(path, "w") as f:
        json.dump(data, f)
