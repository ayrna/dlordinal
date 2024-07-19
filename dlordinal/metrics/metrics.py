import json
import os
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
from sklearn.metrics import confusion_matrix, recall_score


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
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)

    y_oh = np.zeros(y_proba.shape)
    y_oh[np.arange(len(y_true)), y_true] = 1

    y_oh = y_oh.cumsum(axis=1)
    y_proba = y_proba.cumsum(axis=1)

    rps = 0
    for i in range(len(y_true)):
        if y_true[i] in np.arange(y_proba.shape[1]):
            rps += np.power(y_proba[i] - y_oh[i], 2).sum()
        else:
            rps += 1
    return rps / len(y_true)


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
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    sensitivities = recall_score(y_true, y_pred, average=None)
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
            Labels of the classes. If None, the labels are inferred from the data.

    Returns
    -------
    acc : float
            1-off accuracy.

    Examples
    --------
    >>> import numpy as np
    >>> from dlordinal.metrics import accuracy_off1
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 0, 0, 1])
    >>> accuracy_off1(y_true, y_pred)
    0.8571428571428571
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    if labels is None:
        labels = np.unique(y_true)

    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    n = conf_mat.shape[0]
    mask = np.eye(n, n) + np.eye(n, n, k=1), +np.eye(n, n, k=-1)
    correct = mask * conf_mat

    return 1.0 * np.sum(correct) / np.sum(conf_mat)


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
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    sensitivities = recall_score(y_true, y_pred, average=None)
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
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    n_class = cm.shape[0]
    costs = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
    costs = np.abs(costs - np.transpose(costs))
    non_zero_cm_rows = ~np.all(cm == 0, axis=1)
    cm_ = cm[non_zero_cm_rows]
    errors = costs * cm_
    per_class_maes = np.sum(errors, axis=1) / np.sum(cm_, axis=1).astype("double")
    return np.mean(per_class_maes)


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
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    n_class = cm.shape[0]
    costs = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
    costs = np.abs(costs - np.transpose(costs))
    non_zero_cm_rows = ~np.all(cm == 0, axis=1)
    cm_ = cm[non_zero_cm_rows]
    errors = costs * cm_
    per_class_maes = np.sum(errors, axis=1) / np.sum(cm_, axis=1).astype("double")
    return per_class_maes.max()


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
