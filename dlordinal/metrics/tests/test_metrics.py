import json
import os

import numpy as np
import pytest
import torch
from sklearn.metrics import recall_score

from dlordinal.metrics import (
    accuracy_off1,
    amae,
    gmsec,
    minimum_sensitivity,
    mmae,
    ranked_probability_score,
    write_array_to_file,
    write_metrics_dict_to_file,
)


def get_type_convert_lambdas():
    """
    Generate a list of lambda functions that convert input data to different tensor formats.

    This utility function creates multiple lambda functions for testing purposes, each converting
    input data into different tensor representations including numpy arrays, PyTorch tensors,
    and CUDA tensors (if available).

    Returns
    -------
    list of callable
        A list of lambda functions, where each lambda accepts input data and returns it
        converted to a specific format:
        - numpy.ndarray: Converts input to numpy array
        - torch.Tensor: Converts input to PyTorch tensor (CPU)
        - torch.Tensor with requires_grad=True: Converts input to PyTorch tensor with gradient tracking (CPU)
        - torch.Tensor on CUDA: Converts input to PyTorch tensor on GPU (if CUDA available)
        - torch.Tensor with requires_grad=True on CUDA: Converts input to PyTorch tensor with gradient tracking on GPU (if CUDA available)

    Examples
    --------
    >>> lambdas = get_type_convert_lambdas()
    >>> data = [1, 2, 3]
    >>> numpy_result = lambdas[0](data)  # Returns numpy array
    >>> torch_result = lambdas[1](data)  # Returns torch tensor on CPU
    """
    lambdas = []
    # numpy array
    lambdas.append(lambda x: np.array(x))
    # torch tensor
    lambdas.append(lambda x: torch.tensor(x))
    # torch tensor with grad
    lambdas.append(lambda x: torch.tensor(x, requires_grad=True))
    if torch.cuda.is_available():
        # torch tensor on cuda
        lambdas.append(lambda x: torch.tensor(x).cuda())
        # torch tensor with grad on cuda
        lambdas.append(lambda x: torch.tensor(x, requires_grad=True).cuda())
    return lambdas


def run_metric_test(metric_fn, y_true, y_pred, expected_result, rel=1e-6):
    """
    Test a metric function across multiple input data types.

    This function validates that a metric function produces the expected result
    when given different input data type conversions (e.g., list, numpy array, tensor).

    Args:
        metric_fn (callable): The metric function to test. Should accept y_true and y_pred.
        y_true (array-like): Ground truth values.
        y_pred (array-like): Predicted values.
        expected_result (float): The expected metric result.
        rel (float, optional): Relative tolerance for comparison. Defaults to 1e-6.

    Raises:
        AssertionError: If the metric result does not match the expected result
                        within the specified relative tolerance for any type conversion.
    """
    input_lambdas = get_type_convert_lambdas()
    for type_convert_fn in input_lambdas:
        y_t = type_convert_fn(y_true)
        y_p = type_convert_fn(y_pred)
        result = metric_fn(y_t, y_p)
        assert result == pytest.approx(expected_result, rel=rel)


def test_ranked_probability_score():
    y_true = np.array([0, 0, 3, 2])
    y_pred = np.array(
        [
            [0.2, 0.4, 0.2, 0.2],
            [0.7, 0.1, 0.1, 0.1],
            [0.5, 0.05, 0.1, 0.35],
            [0.1, 0.05, 0.65, 0.2],
        ]
    )
    expected_result = 0.506875

    run_metric_test(ranked_probability_score, y_true, y_pred, expected_result)


def test_minimum_sensitivity():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    expected_result = 0.5
    run_metric_test(minimum_sensitivity, y_true, y_pred, expected_result)

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    expected_result = 1.0
    run_metric_test(minimum_sensitivity, y_true, y_pred, expected_result)


def test_accuracy_off1():
    y_true = np.array([0, 1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5, 0])
    expected_result = 8333333333333334
    run_metric_test(accuracy_off1, y_true, y_pred, expected_result)

    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 2, 1, 4, 3])
    expected_result = 1.0
    run_metric_test(accuracy_off1, y_true, y_pred, expected_result)


def test_gmsec():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    sensitivities = recall_score(y_true, y_pred, average=None)
    expected_result = np.sqrt(sensitivities[0] * sensitivities[-1])
    run_metric_test(gmsec, y_true, y_pred, expected_result)

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    sensitivities = recall_score(y_true, y_pred, average=None)
    expected_result = np.sqrt(sensitivities[0] * sensitivities[-1])
    run_metric_test(gmsec, y_true, y_pred, expected_result)


def test_amae():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    expected_result = 0.5
    run_metric_test(amae, y_true, y_pred, expected_result)

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    expected_result = 0.0
    run_metric_test(amae, y_true, y_pred, expected_result)

    y_true = np.array([0, 0, 2, 1])
    y_pred = np.array([0, 2, 0, 1])
    expected_result = 1.0
    run_metric_test(amae, y_true, y_pred, expected_result)

    y_true = np.array([0, 0, 2, 1, 3])
    y_pred = np.array([2, 2, 0, 3, 1])
    expected_result = 2.0
    run_metric_test(amae, y_true, y_pred, expected_result)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    expected_result = 0.5
    run_metric_test(amae, y_true, y_pred, expected_result)

    y_true = np.array([0, 1, 2, 3, 3])
    y_pred = np.array([0, 1, 2, 3, 4])
    expected_result = 0.125
    run_metric_test(amae, y_true, y_pred, expected_result)


def test_mmae():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    expected_result = 0.5
    run_metric_test(mmae, y_true, y_pred, expected_result)

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    expected_result = 0.0
    run_metric_test(mmae, y_true, y_pred, expected_result)

    y_true = np.array([0, 0, 2, 1])
    y_pred = np.array([0, 2, 0, 1])
    expected_result = 2.0
    run_metric_test(mmae, y_true, y_pred, expected_result)

    y_true = np.array([0, 0, 2, 1, 3])
    y_pred = np.array([2, 2, 0, 3, 1])
    expected_result = 2.0
    run_metric_test(mmae, y_true, y_pred, expected_result)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    expected_result = 0.5
    run_metric_test(mmae, y_true, y_pred, expected_result)

    y_true = np.array([0, 1, 2, 3, 3])
    y_pred = np.array([0, 1, 2, 3, 4])
    expected_result = 0.5
    run_metric_test(mmae, y_true, y_pred, expected_result)


def test_write_metrics_dict_to_file():
    metrics = {"acc": 0.5, "gmsec": 0.25}
    path_str = "test_results.txt"
    write_metrics_dict_to_file(metrics, path_str)
    with open(path_str, "r") as f:
        lines = f.read().splitlines()
        assert lines[0] == "acc,gmsec,"
        assert lines[1] == "0.5,0.25,"
    os.remove(path_str)


def test_write_array_to_file():
    array = np.array([0, 1, 2])
    path_str = "test_results.json"
    id_str = "array"
    write_array_to_file(array, path_str, id_str)
    with open(path_str, "r") as f:
        data = json.load(f)
        assert data[id_str] == array.tolist()
    os.remove(path_str)


if __name__ == "__main__":
    test_minimum_sensitivity()
    test_accuracy_off1()
    test_gmsec()
    test_write_metrics_dict_to_file()
    test_write_array_to_file()
