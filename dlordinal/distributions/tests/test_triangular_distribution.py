import numpy as np
import pytest

from ..triangular_distribution import get_triangular_probabilities


def test_get_triangular_probabilities():
    # Case 1
    n = 5
    alpha2 = 0.01
    verbose = 0

    result = get_triangular_probabilities(n, alpha2, verbose)

    expected_result = [
        [0.98845494, 0.01154505, 0.0, 0.0, 0.0],
        [0.01, 0.98, 0.01, 0.0, 0.0],
        [0.0, 0.01, 0.98, 0.01, 0.0],
        [0.0, 0.0, 0.01, 0.98, 0.01],
        [0.0, 0.0, 0.0, 0.00505524, 0.99494475],
    ]

    assert isinstance(result, np.ndarray)
    assert result.shape == (n, n)

    # Sum of probabilities in each row should be approximately 1
    row_sums = np.sum(result, axis=1)
    for row_sum in row_sums:
        assert row_sum == pytest.approx(1.0, abs=1e-6)

    # Individual probabilities should be within [0, 1]
    assert np.all(result >= 0) and np.all(result <= 1)

    # Verfies that result is equal to expected_result with tolerance
    np.testing.assert_allclose(result, expected_result, rtol=1e-6)

    # Case 2
    n = 7
    alpha2 = 0.01
    verbose = 0

    result = get_triangular_probabilities(n, alpha2, verbose)

    expected_result = [
        [0.98845494, 0.01154505, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.01, 0.98, 0.01, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.01, 0.98, 0.01, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.01, 0.98, 0.01, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.01, 0.98, 0.01, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.01, 0.98, 0.01],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.00654182, 0.99345816],
    ]

    assert isinstance(result, np.ndarray)
    assert result.shape == (n, n)

    # Sum of probabilities in each row should be approximately 1
    row_sums = np.sum(result, axis=1)
    for row_sum in row_sums:
        assert row_sum == pytest.approx(1.0, abs=1e-6)

    # Individual probabilities should be within [0, 1]
    assert np.all(result >= 0) and np.all(result <= 1)

    # Verfies that result is equal to expected_result with tolerance
    np.testing.assert_allclose(result, expected_result, rtol=1e-6)


def test_get_triangular_probabilities_verbose():
    # Case 1
    n = 4
    alpha2 = 0.01
    verbose = 4

    result = get_triangular_probabilities(n, alpha2, verbose)

    expected_result = [
        [
            0.98845495,
            0.01154505,
            0.0,
            0.0,
        ],
        [
            0.01,
            0.98,
            0.01,
            0.0,
        ],
        [
            0.0,
            0.01,
            0.98,
            0.01,
        ],
        [0.0, 0.0, 0.00396503, 0.99603496],
    ]

    np.testing.assert_allclose(result, expected_result, rtol=1e-6)
