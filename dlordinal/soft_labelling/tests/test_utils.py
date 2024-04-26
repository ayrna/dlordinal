from dlordinal.soft_labelling.utils import get_intervals, triangular_cdf


def test_get_intervals():
    # Case 1: n = 5
    n = 5
    intervals = get_intervals(n)

    # Check that there are n intervals
    assert len(intervals) == n

    # Check that the intervals are in range [0,1]
    for interval in intervals:
        assert interval[0] >= 0
        assert interval[1] <= 1

    # Check that the intervals are non-overlapping
    for i in range(len(intervals) - 1):
        assert intervals[i][1] == intervals[i + 1][0]


def test_triangular_cdf():
    # Case 1: x <= a
    a = 0.0
    b = 1.0
    c = 0.5
    x = -1.0
    assert triangular_cdf(x, a, b, c) == 0

    # Case 2: a < x < c
    x = 0.25
    expected_result = 0.125
    assert triangular_cdf(x, a, b, c) == expected_result

    # Case 3: c < x < b
    x = 0.75
    expected_result = 0.875
    assert triangular_cdf(x, a, b, c) == expected_result

    # Case 4: b <= x
    x = 1.5
    assert triangular_cdf(x, a, b, c) == 1


if __name__ == "__main__":
    test_get_intervals()
    test_triangular_cdf()
