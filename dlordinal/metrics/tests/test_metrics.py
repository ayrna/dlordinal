import json
import os

import numpy as np
import pytest
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
    assert ranked_probability_score(y_true, y_pred) == pytest.approx(0.506875, rel=1e-6)

    y_true = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    y_pred = np.array(
        [
            [0.2, 0.4, 0.2, 0.2],
            [0.7, 0.1, 0.1, 0.1],
            [0.5, 0.05, 0.1, 0.35],
            [0.1, 0.05, 0.65, 0.2],
        ]
    )
    assert ranked_probability_score(y_true, y_pred) == pytest.approx(0.506875, rel=1e-6)

    y_true = [0, 0, 3, 2]
    y_pred = [
        [0.2, 0.4, 0.2, 0.2],
        [0.7, 0.1, 0.1, 0.1],
        [0.5, 0.05, 0.1, 0.35],
        [0.1, 0.05, 0.65, 0.2],
    ]
    assert ranked_probability_score(y_true, y_pred) == pytest.approx(0.506875, rel=1e-6)

    y_true = [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    y_pred = [
        [0.2, 0.4, 0.2, 0.2],
        [0.7, 0.1, 0.1, 0.1],
        [0.5, 0.05, 0.1, 0.35],
        [0.1, 0.05, 0.65, 0.2],
    ]
    assert ranked_probability_score(y_true, y_pred) == pytest.approx(0.506875, rel=1e-6)


def test_minimum_sensitivity():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    assert minimum_sensitivity(y_true, y_pred) == 0.5

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    assert minimum_sensitivity(y_true, y_pred) == 1.0

    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    assert minimum_sensitivity(y_true, y_pred) == 0.5

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 2, 2]
    assert minimum_sensitivity(y_true, y_pred) == 1.0

    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[0.95, 0.05], [0.1, 0.9], [0.55, 0.45], [0.2, 0.8]])
    assert minimum_sensitivity(y_true, y_pred) == 0.5

    y_true = np.array(
        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
    )
    y_pred = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.15, 0.15],
            [0.0, 0.9, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
            [0.1, 0.2, 0.7],
        ]
    )
    assert minimum_sensitivity(y_true, y_pred) == 1.0

    y_true = [[1, 0], [1, 0], [0, 1], [0, 1]]
    y_pred = [[0.95, 0.05], [0.1, 0.9], [0.55, 0.45], [0.2, 0.8]]
    assert minimum_sensitivity(y_true, y_pred) == 0.5

    y_true = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
    y_pred = [
        [0.8, 0.1, 0.1],
        [0.7, 0.15, 0.15],
        [0.0, 0.9, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
        [0.1, 0.2, 0.7],
    ]
    assert minimum_sensitivity(y_true, y_pred) == 1.0


def test_accuracy_off1():
    y_true = np.array([0, 1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5, 0])
    assert accuracy_off1(y_true, y_pred) == pytest.approx(0.8333333333333334, rel=1e-6)

    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 2, 1, 4, 3])
    assert accuracy_off1(y_true, y_pred) == 1.0

    y_true = [0, 1, 2, 3, 4, 5]
    y_pred = [1, 2, 3, 4, 5, 0]
    assert accuracy_off1(y_true, y_pred) == pytest.approx(0.8333333333333334, rel=1e-6)

    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 2, 1, 4, 3]
    assert accuracy_off1(y_true, y_pred) == 1.0

    y_true = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )
    y_pred = np.array([1, 2, 3, 4, 5, 0])
    assert accuracy_off1(y_true, y_pred) == pytest.approx(0.8333333333333334, rel=1e-6)

    y_true = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )
    y_pred = np.array(
        [
            [0.9, 0.1, 0, 0, 0],
            [0.1, 0.1, 0.8, 0, 0],
            [0, 1.0, 0, 0, 0],
            [0, 0, 0, 0.2, 0.8],
            [0, 0, 0, 0.8, 0.2],
        ]
    )
    assert accuracy_off1(y_true, y_pred) == 1.0

    y_true = [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ]

    y_pred = [1, 2, 3, 4, 5, 0]
    assert accuracy_off1(y_true, y_pred) == pytest.approx(0.8333333333333334, rel=1e-6)

    y_true = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]

    y_pred = [0, 2, 1, 4, 3]
    assert accuracy_off1(y_true, y_pred) == 1.0

    y_true = np.array([0, 1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5, 0])
    assert accuracy_off1(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5]) == pytest.approx(
        0.8333333333333334, rel=1e-6
    )


def test_gmsec():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    result = gmsec(y_true, y_pred)
    sensitivities = recall_score(y_true, y_pred, average=None)
    expected_result = np.sqrt(sensitivities[0] * sensitivities[-1])
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    result = gmsec(y_true, y_pred)
    sensitivities = recall_score(y_true, y_pred, average=None)
    expected_result = np.sqrt(sensitivities[0] * sensitivities[-1])
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    result = gmsec(y_true, y_pred)
    sensitivities = recall_score(y_true, y_pred, average=None)
    expected_result = np.sqrt(sensitivities[0] * sensitivities[-1])
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 2, 2]
    result = gmsec(y_true, y_pred)
    sensitivities = recall_score(y_true, y_pred, average=None)
    expected_result = np.sqrt(sensitivities[0] * sensitivities[-1])
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[0.9, 0.1], [0.1, 0.9], [0.55, 0.45], [0.3, 0.7]])
    result = gmsec(y_true, y_pred)
    sensitivities = recall_score(
        np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), average=None
    )
    expected_result = np.sqrt(sensitivities[0] * sensitivities[-1])
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = np.array(
        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
    )
    y_pred = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.15, 0.15],
            [0.1, 0.8, 0.1],
            [0.2, 0.6, 0.2],
            [0.0, 0.0, 1.0],
            [0.1, 0.2, 0.7],
        ]
    )
    result = gmsec(y_true, y_pred)
    sensitivities = recall_score(
        np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), average=None
    )
    expected_result = np.sqrt(sensitivities[0] * sensitivities[-1])
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = [[1, 0], [1, 0], [0, 1], [0, 1]]
    y_pred = [[0.9, 0.1], [0.1, 0.9], [0.55, 0.45], [0.3, 0.7]]
    result = gmsec(y_true, y_pred)
    sensitivities = recall_score(
        np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), average=None
    )
    expected_result = np.sqrt(sensitivities[0] * sensitivities[-1])
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
    y_pred = [
        [0.8, 0.1, 0.1],
        [0.7, 0.15, 0.15],
        [0.1, 0.8, 0.1],
        [0.2, 0.6, 0.2],
        [0.0, 0.0, 1.0],
        [0.1, 0.2, 0.7],
    ]
    result = gmsec(y_true, y_pred)
    sensitivities = recall_score(
        np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), average=None
    )
    expected_result = np.sqrt(sensitivities[0] * sensitivities[-1])
    assert result == pytest.approx(expected_result, rel=1e-6)


def test_amae():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    result = amae(y_true, y_pred)
    expected_result = 0.5
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    result = amae(y_true, y_pred)
    expected_result = 0.0
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = np.array([0, 0, 2, 1])
    y_pred = np.array([0, 2, 0, 1])
    result = amae(y_true, y_pred)
    expected_result = 1.0
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = np.array([0, 0, 2, 1, 3])
    y_pred = np.array([2, 2, 0, 3, 1])
    result = amae(y_true, y_pred)
    expected_result = 2.0
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    result = amae(y_true, y_pred)
    expected_result = 0.5
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 2, 2]
    result = amae(y_true, y_pred)
    expected_result = 0.0
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = [0, 0, 2, 1]
    y_pred = [0, 2, 0, 1]
    result = amae(y_true, y_pred)
    expected_result = 1.0
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = [0, 0, 2, 1, 3]
    y_pred = [2, 2, 0, 3, 1]
    result = amae(y_true, y_pred)
    expected_result = 2.0
    assert result == pytest.approx(expected_result, rel=1e-6)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    result = amae(y_true, y_pred)
    expected_result = 0.5
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = [[1, 0], [1, 0], [0, 1], [0, 1]]
    y_pred = [[1, 0], [0, 1], [1, 0], [0, 1]]
    result = amae(y_true, y_pred)
    expected_result = 0.5
    assert result == pytest.approx(expected_result, rel=1e-6)


def test_mmae():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    result = mmae(y_true, y_pred)
    expected_result = 0.5
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    result = mmae(y_true, y_pred)
    expected_result = 0.0
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = np.array([0, 0, 2, 1])
    y_pred = np.array([0, 2, 0, 1])
    result = mmae(y_true, y_pred)
    expected_result = 2.0
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = np.array([0, 0, 2, 1, 3])
    y_pred = np.array([2, 2, 0, 3, 1])
    result = mmae(y_true, y_pred)
    expected_result = 2.0
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    result = mmae(y_true, y_pred)
    expected_result = 0.5
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 2, 2]
    result = mmae(y_true, y_pred)
    expected_result = 0.0
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = [0, 0, 2, 1]
    y_pred = [0, 2, 0, 1]
    result = mmae(y_true, y_pred)
    expected_result = 2.0
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = [0, 0, 2, 1, 3]
    y_pred = [2, 2, 0, 3, 1]
    result = mmae(y_true, y_pred)
    expected_result = 2.0
    assert result == pytest.approx(expected_result, rel=1e-6)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    result = mmae(y_true, y_pred)
    expected_result = 0.5
    assert result == pytest.approx(expected_result, rel=1e-6)

    y_true = [[1, 0], [1, 0], [0, 1], [0, 1]]
    y_pred = [[1, 0], [0, 1], [1, 0], [0, 1]]
    result = mmae(y_true, y_pred)
    expected_result = 0.5
    assert result == pytest.approx(expected_result, rel=1e-6)


def test_write_metrics_dict_to_file():
    metrics = {"acc": 0.5, "gmsec": 0.25}
    path_str = "test_results.txt"
    write_metrics_dict_to_file(metrics, path_str)
    with open(path_str, "r") as f:
        lines = f.read().splitlines()
        assert lines[0] == "acc,gmsec,"
        assert lines[1] == "0.5,0.25,"
    os.remove(path_str)


def test_write_array_to_file(tmp_path):
    array = np.array([0, 1, 2])
    path_str = tmp_path / "test_results.json"
    id_str = "array"
    write_array_to_file(array, path_str, id_str)
    with open(path_str, "r") as f:
        data = json.load(f)
        assert data[id_str] == array.tolist()
