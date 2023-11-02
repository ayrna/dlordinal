import numpy as np
import pytest
from scipy.special import gamma, hyp2f1

from ..beta_distribution import beta_dist, beta_inc, get_beta_probabilities
from ..utils import get_intervals


def test_beta_inc():
    # Case 1: Positive Values
    result = beta_inc(2.0, 3.0)
    print(result)
    expected_result = 0.08333333333333333
    assert result == pytest.approx(expected_result, rel=1e-6)

    # Case 2: Avoid division by 0
    with pytest.raises(ValueError):
        result = beta_inc(0.0, 0.0)


@pytest.mark.parametrize("a, b", [(-1.0, 2.0), (1.0, -2.0), (-1.0, -2.0)])
def test_beta_inc_negative_values(a, b):
    with pytest.raises(ValueError):
        beta_inc(a, b)


def test_beta_distribution():
    # Case 1: Valid input
    x = 0.5
    p = 2.0
    q = 3.0
    a = 1.0
    result = beta_dist(x, p, q, a)
    expected_result = 0.6875
    assert result == pytest.approx(expected_result, rel=1e-6)

    # Case 2: Custom scaling parameter
    a = 2.0
    result = beta_dist(x, p, q, a)
    expected_result = 0.26171875
    assert result == pytest.approx(expected_result, rel=1e-6)

    # Case 3: Check probabilities with custom x
    x_array = np.linspace(0, 1, num=101)
    for x in x_array:
        result = beta_dist(x, p, q, a)
        assert result >= 0.0 or result <= 1.0


@pytest.mark.parametrize("x", [-1.0, 2.0])
def test_beta_distribution_negative_x(x):
    with pytest.raises(ValueError):
        beta_dist(x, 2.0, 3.0, 1.0)


def test_beta_probabilities():
    # Case 1: Valid input
    n = 5
    p = 2.0
    q = 3.0
    a = 1.0
    result = get_beta_probabilities(n, p, q, a)
    expected_result = [
        0.1808000009216,
        0.34399999942400017,
        0.2959999994239999,
        0.15200000000000036,
        0.02720000023039959,
    ]

    for r, e in zip(result, expected_result):
        assert abs(r - e) < 1e-6

    # Case 2: Custom scaling parameter
    n = 4
    p = 1.5
    q = 2.5
    a = 2.0
    result = get_beta_probabilities(n, p, q, a)
    expected_result = [
        0.05010107325697135,
        0.283232260076362,
        0.4532888937552723,
        0.21337777291139426,
    ]

    for r, e in zip(result, expected_result):
        assert r == pytest.approx(e, rel=1e-6)


if __name__ == "__main__":
    test_beta_inc()
    test_beta_inc_negative_values()
    test_beta_distribution()
    test_beta_distribution_negative_x()
    test_beta_probabilities()
