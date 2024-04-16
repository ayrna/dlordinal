import re

import numpy as np
import pytest

from dlordinal.soft_labelling import (
    get_general_triangular_params,
    get_general_triangular_soft_labels,
)


def test_get_general_triangular_params():
    # Case 1
    n = 3
    alphas = np.array(
        [0.05518804, 0.14000449, 0.0586412, 0.03018706, 0.15230179, 0.03493327]
    )
    params = get_general_triangular_params(n, alphas)

    # Verifies that the number of generated parameters is equal to n
    assert len(params) == n

    # Verifies the values of the parameters for each class
    expected_params = [
        {"alpha2j_1": 0, "alpha2j": 0.0586412, "a": 0, "b": 0.4398462632248991, "c": 0},
        {
            "alpha2j_1": 0.03018706,
            "alpha2j": 0.15230179,
            "a": 0.270712925594274,
            "b": 0.837253986265032,
            "c": 0.5,
        },
        {
            "alpha2j_1": 0.03493327,
            "alpha2j": 0,
            "a": 0.5900440857512113,
            "b": 1,
            "c": 1,
        },
    ]

    # Verifies that the values of the parameters are equal to the expected values
    for i in range(n):
        assert params[i] == pytest.approx(expected_params[i], rel=1e-6)

    # Case 2
    n = 4
    alphas = np.array(
        [
            0.17519177,
            0.04555998,
            0.03995811,
            0.0118866,
            0.0563474,
            0.17944881,
            0.19793265,
            0.19175102,
        ]
    )
    params = get_general_triangular_params(n, alphas)

    # Verifies that the number of generated parameters is equal to n
    assert len(params) == n

    # Verifies the values of the parameters for each class
    expected_params = [
        {
            "alpha2j_1": 0,
            "alpha2j": 0.03995811,
            "a": 0,
            "b": 0.3124590864382281,
            "c": 0,
        },
        {
            "alpha2j_1": 0.0118866,
            "alpha2j": 0.0563474,
            "a": 0.225688305781725,
            "b": 0.558714947376686,
            "c": 0.375,
        },
        {
            "alpha2j_1": 0.17944881,
            "alpha2j": 0.19793265,
            "a": 0.307987644692798,
            "b": 0.956086983327608,
            "c": 0.625,
        },
        {
            "alpha2j_1": 0.19175102,
            "alpha2j": 0,
            "a": 0.5552441508854093,
            "b": 1,
            "c": 1,
        },
    ]

    # Verifies that the values of the parameters are equal to the expected values
    for i in range(n):
        assert params[i] == pytest.approx(expected_params[i], rel=1e-6)


def test_wrong_alpha_shape():
    n = 3
    alphas = np.array(
        [
            0.05518804,
            0.14000449,
            0.0586412,
            0.03018706,
            0.15230179,
            0.03493327,
            0.19175102,
        ]
    )

    with pytest.raises(
        ValueError,
        match=re.escape("alphas must be a numpy array of shape (2 * n,), but got (7,)"),
    ):
        get_general_triangular_params(n, alphas)


def test_general_triangular_soft_labels():
    n = 3
    alphas = np.array(
        [0.05518804, 0.14000449, 0.0586412, 0.03018706, 0.15230179, 0.03493327]
    )
    result = get_general_triangular_soft_labels(n, alphas)
    expected_result = [
        [0.9413588, 0.0586412, 0.0],
        [0.03018706, 0.81751114, 0.15230179],
        [0.0, 0.03493327, 0.96506673],
    ]

    assert len(result.shape) == 2
    assert result.shape[0] == n
    assert result.shape[1] == n

    assert np.allclose(result, expected_result, rtol=1e-6)
