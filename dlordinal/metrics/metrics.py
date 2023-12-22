import numpy as np
from sklearn.metrics import confusion_matrix
from pathlib import Path
from typing import Callable, Dict, Optional
import json
import os
from sklearn.metrics import recall_score


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
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0, 1, 0, 1])
    >>> minimum_sensitivity(y_true, y_pred)
    0.5
    """

    sensitivities = recall_score(y_true, y_pred, average=None)
    return np.min(sensitivities)


def accuracy_off1(y_true: np.ndarray, y_pred: np.ndarray, labels=None) -> float:
    """Computes the accuracy of the predictions, allowing errors if they occur in an adjacent class.

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
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0, 1, 0, 1])
    >>> accuracy_off1(y_true, y_pred)
    1.0
    """

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
    """Geometric mean of the sensitivity of the extreme classes.
    Determines how good the classification performance for the first and the last
    classes is.

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
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0, 1, 0, 1])
    >>> gmec(y_true, y_pred)
    0.5
    """

    sensitivities = recall_score(y_true, y_pred, average=None)
    return np.sqrt(sensitivities[0] * sensitivities[-1])


def write_metrics_dict_to_file(
    metrics: Dict[str, float],
    path_str: str,
    filter_fn: Optional[Callable[[str, float], bool]] = None,
) -> None:
    """Writes a dictionary of metrics to a tabular file.
    The dictionary is filtered by the filter function.
    The first time that the metrics are saved to the file, the keys are written as the header.
    Subsequent calls append the values to the file.

    Parameters
    ----------
    metrics : Dict[str, float]
            Dictionary of metric names associated with their value.
    path_str : str
            Path to the file that will be saved.
            The directory of the file will be created if it does not exist.
            If the file exists, the metrics will be appended to the file in a new row.
    filter_fn : Optional[Callable[[str, bool], bool]]
            Function that filters the metrics.
            The function takes the name and the value of the metric and returns ``True`` if the metric should be saved.

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

    filter_fn: Callable[[str, bool], bool] = (
        filter_fn if filter_fn is not None else lambda n, v: True
    )
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
