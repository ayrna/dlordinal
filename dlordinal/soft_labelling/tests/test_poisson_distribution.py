import numpy as np
import pytest

from dlordinal.soft_labelling import get_poisson_soft_labels


def test_get_poisson_soft_labels():
    # Case 1: n = 3
    n = 3
    result = get_poisson_soft_labels(n)

    # Verifies that the result is a matrix with n rows and n columns
    assert result.shape == (n, n)

    expected_probs = [
        [0.3531091, 0.3531091, 0.29378181],
        [0.30396604, 0.34801698, 0.34801698],
        [0.30348469, 0.33525965, 0.36125566],
    ]

    # Verifies that the generated probabilities are equal to the expected ones with
    # tolerance
    np.testing.assert_allclose(result, expected_probs, rtol=1e-6)

    # Sum of probabilities in each row should be approximately 1
    row_sums = np.sum(result, axis=1)
    for row_sum in row_sums:
        assert row_sum == pytest.approx(1.0, abs=1e-6)

    # Individual probabilities should be within [0, 1]
    assert np.all(result >= 0) and np.all(result <= 1)

    # Case 2: n = 5
    n = 5
    result = get_poisson_soft_labels(n)

    print(result)
    # Verifies that the result is a matrix with n rows and n columns
    assert result.shape == (n, n)

    expected_probs = [
        [0.23414552, 0.23414552, 0.19480578, 0.17232403, 0.16457916],
        [0.18896888, 0.21635436, 0.21635436, 0.19768881, 0.18063359],
        [0.17822335, 0.19688341, 0.21214973, 0.21214973, 0.20059378],
        [0.17919028, 0.18931175, 0.20370191, 0.21389803, 0.21389803],
        [0.18400408, 0.18903075, 0.19882883, 0.21031236, 0.21782399],
    ]

    # Verifies that the generated probabilities are equal to the expected ones with
    # tolerance
    np.testing.assert_allclose(result, expected_probs, rtol=1e-6)

    # Sum of probabilities in each row should be approximately 1
    row_sums = np.sum(result, axis=1)
    for row_sum in row_sums:
        assert row_sum == pytest.approx(1.0, abs=1e-6)

    # Individual probabilities should be within [0, 1]
    assert np.all(result >= 0) and np.all(result <= 1)
