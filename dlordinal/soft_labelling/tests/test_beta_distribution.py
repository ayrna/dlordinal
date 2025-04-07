import numpy as np
import pytest

from dlordinal.soft_labelling import get_beta_soft_labels
from dlordinal.soft_labelling.beta_distribution import (
    _beta_dist,
    _beta_func,
    _get_beta_soft_label,
)


def test_beta_inc():
    # Case 1: Positive Values
    result = _beta_func(2.0, 3.0)
    print(result)
    expected_result = 0.08333333333333333
    assert result == pytest.approx(expected_result, rel=1e-6)

    # Case 2: Avoid division by 0
    with pytest.raises(ValueError):
        result = _beta_func(0.0, 0.0)


@pytest.mark.parametrize("a, b", [(-1.0, 2.0), (1.0, -2.0), (-1.0, -2.0)])
def test_beta_inc_negative_values(a, b):
    with pytest.raises(ValueError):
        _beta_func(a, b)


def test_beta_dist():
    # Case 1: Valid input
    x = 0.5
    p = 2.0
    q = 3.0
    a = 1.0
    result = _beta_dist(x, p, q, a)
    expected_result = 0.6875
    assert result == pytest.approx(expected_result, rel=1e-6)

    # Case 2: Custom scaling parameter
    a = 2.0
    result = _beta_dist(x, p, q, a)
    expected_result = 0.26171875
    assert result == pytest.approx(expected_result, rel=1e-6)

    # Case 3: Check probabilities with custom x
    x_array = np.linspace(0, 1, num=101)
    for x in x_array:
        result = _beta_dist(x, p, q, a)
        assert result >= 0.0 or result <= 1.0


@pytest.mark.parametrize("x", [-1.0, 2.0])
def test_beta_distribution_negative_x(x):
    with pytest.raises(ValueError):
        _beta_dist(x, 2.0, 3.0, 1.0)


def test_beta_soft_label():
    # Case 1: Valid input
    n = 5
    p = 2.0
    q = 3.0
    a = 1.0
    result = _get_beta_soft_label(n, p, q, a)
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
    result = _get_beta_soft_label(n, p, q, a)
    expected_result = [
        0.05010107325697135,
        0.283232260076362,
        0.4532888937552723,
        0.21337777291139426,
    ]

    for r, e in zip(result, expected_result):
        assert r == pytest.approx(e, rel=1e-6)


def test_beta_softlabels():
    n = 5
    result = get_beta_soft_labels(n)

    assert len(result.shape) == 2
    assert result.shape[0] == n
    assert result.shape[1] == n

    expected_result = [
        [
            0.8322278330066323,
            0.15097599903815717,
            0.01614079995258888,
            0.0006528000025611824,
            2.5600000609360407e-06,
        ],
        [
            0.16306230596685573,
            0.6740152119811103,
            0.15985465217901373,
            0.003067150177798128,
            6.796952229937148e-07,
        ],
        [
            0.0005973937258183486,
            0.16304604740785777,
            0.6727131177326486,
            0.16304604740785367,
            0.000597393725820794,
        ],
        [
            6.796952207410873e-07,
            0.0030671501777993896,
            0.15985465217901168,
            0.6740152119811109,
            0.16306230596685678,
        ],
        [
            2.560000061440001e-06,
            0.0006528000025600005,
            0.01614079995258879,
            0.1509759990381568,
            0.8322278330066332,
        ],
    ]

    for r, e in zip(result, expected_result):
        for r_, e_ in zip(r, e):
            assert r_ == pytest.approx(e_, rel=1e-6)


def test_beta_softlabels_invalid_params():
    with pytest.raises(ValueError):
        get_beta_soft_labels(0)

    with pytest.raises(ValueError):
        get_beta_soft_labels(3, params_set="invalid")

    with pytest.raises(ValueError):
        get_beta_soft_labels(3, params_set={"test": [[1, 2, 3]]})

    with pytest.raises(ValueError):
        get_beta_soft_labels(3, params_set={4: [[1, 2, 3]]})

    with pytest.raises(ValueError):
        get_beta_soft_labels(3, params_set={3: [[1, 2]]})

    with pytest.raises(ValueError):
        get_beta_soft_labels(3, params_set={3: [[1, 2, 3], [4, 5]]})

    with pytest.raises(ValueError):
        get_beta_soft_labels(3, params_set={3: [[1, 2, 3], [4, 5, 6], [7]]})

    with pytest.raises(ValueError):
        get_beta_soft_labels(4, params_set={4: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]})

    with pytest.raises(ValueError):
        get_beta_soft_labels(4, params_set=10)


@pytest.mark.parametrize("J", [3, 4, 5, 6, 7, 8, 9, 10])
def test_beta_softlabels_custom_params(J):
    params = {
        3: [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        4: [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        5: [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
        6: [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
        7: [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
        ],
        8: [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 24],
        ],
        9: [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 24],
            [25, 26, 27],
        ],
        10: [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 24],
            [25, 26, 27],
            [28, 29, 30],
        ],
    }
    result = get_beta_soft_labels(J, params_set=params)

    assert result.ndim == 2
    assert result.shape == (J, J)


@pytest.mark.parametrize("J", [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
def test_beta_softlabels_custom_params_2(J):
    from dlordinal.soft_labelling.beta_distribution import _beta_params_sets

    params = _beta_params_sets["standard"]
    result_standard = get_beta_soft_labels(J, params_set="standard")
    result_custom = get_beta_soft_labels(J, params_set=params)
    assert result_standard == pytest.approx(result_custom, rel=1e-6)
