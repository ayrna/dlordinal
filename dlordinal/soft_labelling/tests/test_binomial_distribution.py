import numpy as np
import pytest

from dlordinal.soft_labelling import get_binomial_soft_labels


def test_get_binomial_soft_labels():
    # Case 1: n = 5
    n = 5
    result = get_binomial_soft_labels(n)
    expected_result = np.array(
        [
            [6.561e-01, 2.916e-01, 4.860e-02, 3.600e-03, 1.000e-04],
            [2.401e-01, 4.116e-01, 2.646e-01, 7.560e-02, 8.100e-03],
            [6.250e-02, 2.500e-01, 3.750e-01, 2.500e-01, 6.250e-02],
            [8.100e-03, 7.560e-02, 2.646e-01, 4.116e-01, 2.401e-01],
            [1.000e-04, 3.600e-03, 4.860e-02, 2.916e-01, 6.561e-01],
        ]
    )

    # Compare the probability matrices with a tolerance of 1e-3
    np.testing.assert_allclose(result, expected_result, rtol=1e-3)

    # Sum of probabilities in each row should be approximately 1
    row_sums = np.sum(result, axis=1)
    for row_sum in row_sums:
        assert row_sum == pytest.approx(1.0, abs=1e-6)

    # Individual probabilities should be within [0, 1]
    assert np.all(result >= 0) and np.all(result <= 1)


def test_get_binomial_soft_labels_invalid_input():
    with pytest.raises(ValueError):
        get_binomial_soft_labels(1)

    with pytest.raises(ValueError):
        get_binomial_soft_labels(1.0)

    with pytest.raises(ValueError):
        get_binomial_soft_labels(0)

    with pytest.raises(ValueError):
        get_binomial_soft_labels(-1)


def test_get_binomial_soft_labels_valid_input():
    for i in range(2, 11):
        soft_labels = get_binomial_soft_labels(i)

        # Sum of probabilities in each row should be approximately 1
        row_sums = np.sum(soft_labels, axis=1)
        for row_sum in row_sums:
            assert row_sum == pytest.approx(1.0, abs=1e-6)

        # Check that all the elements in the matrix are less than or equal to the
        # element in the diagonal
        diagonal = np.diag(soft_labels)
        diff = soft_labels - diagonal[:, np.newaxis]
        assert np.all(diff <= 1e-9)
