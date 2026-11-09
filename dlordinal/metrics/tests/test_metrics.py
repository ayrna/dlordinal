import json
import os
from math import sqrt

import numpy as np
import pytest
import torch
from sklearn.metrics import recall_score

from dlordinal.metrics import (
    accuracy_off1,
    amae,
    gmes,
    gmsec,
    mes,
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
    lambdas.append(lambda x: torch.tensor(x, dtype=torch.float32, requires_grad=True))
    if torch.cuda.is_available():
        # torch tensor on cuda
        lambdas.append(lambda x: torch.tensor(x).cuda())
        # torch tensor with grad on cuda
        lambdas.append(
            lambda x: torch.tensor(x, dtype=torch.float32, requires_grad=True).cuda()
        )
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
    expected_result = 0.8333333333333334
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


def test_mes():
    # arithmetic mean of extreme class sensitivities
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    sensitivities = recall_score(y_true, y_pred, average=None)
    expected_result = (sensitivities[0] + sensitivities[-1]) / 2.0
    run_metric_test(mes, y_true, y_pred, expected_result)

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    sensitivities = recall_score(y_true, y_pred, average=None)
    expected_result = (sensitivities[0] + sensitivities[-1]) / 2.0
    run_metric_test(mes, y_true, y_pred, expected_result)


def test_gmes():
    # geometric mean of extreme class sensitivities
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    sensitivities = recall_score(y_true, y_pred, average=None)
    expected_result = sqrt(sensitivities[0] * sensitivities[-1])
    run_metric_test(gmes, y_true, y_pred, expected_result)

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    sensitivities = recall_score(y_true, y_pred, average=None)
    expected_result = sqrt(sensitivities[0] * sensitivities[-1])
    run_metric_test(gmes, y_true, y_pred, expected_result)


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


def test_ranked_probability_score_matches_per_sample_definition():
    rng = np.random.default_rng(0)
    y_proba = rng.dirichlet(np.ones(5), size=200)
    y_true = rng.integers(0, 5, size=200)
    expected = np.mean(
        [
            np.sum((np.cumsum(p) - np.cumsum(np.eye(5)[y])) ** 2)
            for y, p in zip(y_true, y_proba)
        ]
    )
    assert ranked_probability_score(y_true, y_proba) == pytest.approx(expected)


def test_ranked_probability_score_rejects_out_of_range_labels():
    y_proba = np.full((2, 3), 1 / 3)
    with pytest.raises(ValueError):
        ranked_probability_score(np.array([0, 3]), y_proba)
    with pytest.raises(ValueError):
        ranked_probability_score(np.array([0, -1]), y_proba)


def test_accuracy_off1_classes_absent_from_y_true():
    # Class 1 is absent from y_true: classes 0 and 2 must not count as adjacent
    run_metric_test(accuracy_off1, [0, 2, 2], [2, 2, 2], 2 / 3)
    # The prediction 3 is absent from y_true: it must still count as an error
    run_metric_test(accuracy_off1, [0, 0, 1], [0, 0, 3], 2 / 3)


def test_accuracy_off1_explicit_labels():
    labels = [0, 1, 2, 3]
    assert accuracy_off1([0, 2, 2], [2, 2, 2], labels=labels) == pytest.approx(2 / 3)
    # Adjacency follows the order of the labels list
    assert accuracy_off1(["a", "c"], ["b", "b"], labels=["a", "b", "c"]) == 1.0
    with pytest.raises(ValueError):
        accuracy_off1([0, 1], [0, 5], labels=labels)


def test_amae_mmae_use_label_distances():
    # Class 2 is absent from both arrays: the error 1 -> 3 is 2, not 1
    y_true = np.array([0, 1, 3, 3])
    y_pred = np.array([0, 3, 3, 3])
    run_metric_test(amae, y_true, y_pred, 2 / 3)
    run_metric_test(mmae, y_true, y_pred, 2.0)


def test_sensitivity_metrics_ignore_labels_absent_from_y_true():
    # The prediction 2 never occurs in y_true and has no sensitivity
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 2])
    run_metric_test(minimum_sensitivity, y_true, y_pred, 0.5)
    run_metric_test(gmsec, y_true, y_pred, np.sqrt(0.5))
    run_metric_test(mes, y_true, y_pred, 0.75)
    run_metric_test(gmes, y_true, y_pred, np.sqrt(0.5))


@pytest.mark.parametrize("metric", [mes, gmes, gmsec, minimum_sensitivity])
def test_sensitivity_metrics_accept_probabilities(metric):
    y_true = np.array([0, 1, 2])
    y_proba = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    run_metric_test(metric, y_true, y_proba, 1.0)
