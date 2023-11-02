import numpy as np
import pytest

from ..exponential_distribution import get_exponential_probabilities


def test_get_exponential_probabilities():
    n = 5
    p = 1.0
    tau = 1.0
    result = get_exponential_probabilities(n, p, tau)
    expected_result = np.array(
        [
            [0.63640865, 0.23412166, 0.08612854, 0.03168492, 0.01165623],
            [0.19151597, 0.52059439, 0.19151597, 0.07045479, 0.02591887],
            [0.06745081, 0.1833503, 0.49839779, 0.1833503, 0.06745081],
            [0.02591887, 0.07045479, 0.19151597, 0.52059439, 0.19151597],
            [0.01165623, 0.03168492, 0.08612854, 0.23412166, 0.63640865],
        ]
    )

    # Compare the probability matrices with a tolerance of 1e-3
    np.testing.assert_allclose(result, expected_result, rtol=1e-6)

    # Sum of probabilities in each row should be approximately 1
    row_sums = np.sum(result, axis=1)
    for row_sum in row_sums:
        assert row_sum == pytest.approx(1.0, abs=1e-6)

    # Individual probabilities should be within [0, 1]
    assert np.all(result >= 0) and np.all(result <= 1)


if __name__ == "__main__":
    test_get_exponential_probabilities()
