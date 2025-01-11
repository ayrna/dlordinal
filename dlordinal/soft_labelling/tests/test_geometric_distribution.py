import re

import numpy as np
import pytest

from dlordinal.soft_labelling import get_geometric_soft_labels


@pytest.mark.parametrize("n, alphas", [(-1, 0.1), (0, 0.1), (1, 0.1)])
def test_get_geometric_soft_labels_number_of_classes_error(n, alphas):
    with pytest.raises(
        ValueError, match=f"J={n} must be a positive integer greater than 1"
    ):
        get_geometric_soft_labels(n, alphas)


@pytest.mark.parametrize(
    "n, alphas",
    [
        (3, -0.1),
        (3, 1.1),
        (3, [0.1, 0.1, -0.1]),
        (3, [1.1, 0.1, 0.1]),
        (3, [(1.1, 0.5, 0.5), (0.1, 0.5, 0.5), (0.1, 0.5, 0.5)]),
    ],
)
def test_get_geometric_soft_labels_alphas_range_error(n, alphas):
    with pytest.raises(
        ValueError, match=re.escape(f"alphas={alphas} must be in the range [0, 1]")
    ):
        get_geometric_soft_labels(n, alphas)


@pytest.mark.parametrize("n, alphas", [(3, [0.1, 0.2])])
def test_get_geometric_soft_labels_alphas_list_size_error(n, alphas):
    with pytest.raises(
        ValueError, match=f"Size of alphas={len(alphas)} must be equal to J={n}"
    ):
        get_geometric_soft_labels(n, alphas)


@pytest.mark.parametrize("n, alphas", [(3, None), (3, ["0.1", "0.2", "0.1"])])
def test_get_geometric_soft_labels_alphas_wrong_type_error(n, alphas):
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"alphas={alphas} must either be a single float value [0,1],"
            " a list of floats, or tuples of the form (alpha,F_l,F_r)"
        ),
    ):
        get_geometric_soft_labels(n, alphas)


@pytest.mark.parametrize(
    "n, alphas", [(3, [(0.1, 0.6, 0.5), (0.1, 0.5, 0.5), (0.1, 0.5, 0.5)])]
)
def test_get_geometric_soft_labels_alphas_wrong_fractions(n, alphas):
    with pytest.raises(
        ValueError, match=re.escape(f"F_l and F_r must sum to one, alphas={alphas}")
    ):
        get_geometric_soft_labels(n, alphas)


def test_get_geometric_soft_labels_single_alpha_default():
    # Arrange
    n = 5
    alpha = 0.1
    expected_result = np.array(
        [
            [
                9.00000000e-01,
                9.00090009e-02,
                9.00090009e-03,
                9.00090009e-04,
                9.00090009e-05,
            ],
            [
                4.73933649e-02,
                9.00000000e-01,
                4.73933649e-02,
                4.73933649e-03,
                4.73933649e-04,
            ],
            [
                4.54545455e-03,
                4.54545455e-02,
                9.00000000e-01,
                4.54545455e-02,
                4.54545455e-03,
            ],
            [
                4.73933649e-04,
                4.73933649e-03,
                4.73933649e-02,
                9.00000000e-01,
                4.73933649e-02,
            ],
            [
                9.00090009e-05,
                9.00090009e-04,
                9.00090009e-03,
                9.00090009e-02,
                9.00000000e-01,
            ],
        ]
    )

    # Act
    result = get_geometric_soft_labels(n)

    # Assert that unimodal smoothings for each class sum to one
    np.testing.assert_equal(np.mean(np.sum(result, axis=1)), 1.0)
    # Assert that true class has 1 - alpha probability mass
    np.testing.assert_equal(np.diagonal(result), 1.0 - alpha)
    # Assert exact match
    np.testing.assert_allclose(expected_result, result)


def test_get_geometric_soft_labels_single_alpha():
    # Arrange
    n = 5
    alpha = 0.05
    expected_result = np.array(
        [
            [
                9.50000000e-01,
                4.75002969e-02,
                2.37501484e-03,
                1.18750742e-04,
                5.93753711e-06,
            ],
            [
                2.43605359e-02,
                9.50000000e-01,
                2.43605359e-02,
                1.21802680e-03,
                6.09013398e-05,
            ],
            [
                1.19047619e-03,
                2.38095238e-02,
                9.50000000e-01,
                2.38095238e-02,
                1.19047619e-03,
            ],
            [
                6.09013398e-05,
                1.21802680e-03,
                2.43605359e-02,
                9.50000000e-01,
                2.43605359e-02,
            ],
            [
                5.93753711e-06,
                1.18750742e-04,
                2.37501484e-03,
                4.75002969e-02,
                9.50000000e-01,
            ],
        ]
    )

    # Act
    result = get_geometric_soft_labels(n, alpha)

    # Assert that unimodal smoothings for each class sum to one
    np.testing.assert_equal(np.mean(np.sum(result, axis=1)), 1.0)
    # Assert that true class has 1 - alpha probability mass
    np.testing.assert_equal(np.diagonal(result), 1.0 - alpha)
    # Assert exact match
    np.testing.assert_allclose(expected_result, result)


def test_get_geometric_soft_labels_list():
    # Arrange
    n = 5
    alphas = [0.2, 0.1, 0.05, 0.1, 0.2]
    expected_result = np.array(
        [
            [
                8.00000000e-01,
                1.60256410e-01,
                3.20512821e-02,
                6.41025641e-03,
                1.28205128e-03,
            ],
            [
                4.73933649e-02,
                9.00000000e-01,
                4.73933649e-02,
                4.73933649e-03,
                4.73933649e-04,
            ],
            [
                1.19047619e-03,
                2.38095238e-02,
                9.50000000e-01,
                2.38095238e-02,
                1.19047619e-03,
            ],
            [
                4.73933649e-04,
                4.73933649e-03,
                4.73933649e-02,
                9.00000000e-01,
                4.73933649e-02,
            ],
            [
                1.28205128e-03,
                6.41025641e-03,
                3.20512821e-02,
                1.60256410e-01,
                8.00000000e-01,
            ],
        ]
    )

    # Act
    result = get_geometric_soft_labels(n, alphas)

    # Assert that unimodal smoothings for each class sum to one
    np.testing.assert_equal(np.mean(np.sum(result, axis=1)), 1.0)
    # Assert that true class has 1 - alpha probability mass
    np.testing.assert_array_equal(np.diagonal(result), 1.0 - np.array(alphas))
    # Assert exact match
    np.testing.assert_allclose(expected_result, result)


def test_get_geometric_soft_labels_list_with_zeros():
    # Arrange
    n = 5
    alphas = [0.0, 0.1, 0.05, 0.1, 0.0]

    # Act
    result = get_geometric_soft_labels(n, alphas)

    # Assert that unimodal smoothings for each class sum to one
    np.testing.assert_equal(np.mean(np.sum(result, axis=1)), 1.0)
    # Assert that true class has 1 - alpha probability mass
    np.testing.assert_array_equal(np.diagonal(result), 1.0 - np.array(alphas))


def test_get_geometric_soft_labels_relations_with_zeros():
    # Arrange
    n = 5
    alphas = [
        (0.0, 0.0, 1.0),
        (0.1, 0.5, 0.5),
        (0.05, 0.6, 0.4),
        (0.1, 0.4, 0.6),
        (0.0, 1.0, 0.0),
    ]

    # Act
    result = get_geometric_soft_labels(n, alphas)

    # Assert that unimodal smoothings for each class sum to one
    np.testing.assert_equal(np.mean(np.sum(result, axis=1)), 1.0)
    # Assert that true class has 1 - alpha probability mass
    np.testing.assert_array_equal(
        np.diagonal(result), 1.0 - np.array([alpha[0] for alpha in alphas])
    )


def test_get_geometric_soft_labels_asymmetric_a():
    # Arrange
    n = 5
    alphas = [(0.2, 0, 1), (0.1, 0.5, 0.5), (0.05, 1, 0), (0.1, 0, 1), (0.2, 1, 0)]

    # Act
    result = get_geometric_soft_labels(n, alphas)

    # Assert that unimodal smoothings for each class sum to one
    np.testing.assert_equal(np.mean(np.sum(result, axis=1)), 1.0)
    # Assert that true class has 1 - alpha probability mass
    np.testing.assert_array_equal(
        np.diagonal(result), 1.0 - np.array([alpha[0] for alpha in alphas])
    )


def test_get_geometric_soft_labels_asymmetric_b():
    # Arrange
    n = 5
    alphas = [
        (0.2, 0.5, 0.5),
        (0.1, 0.5, 0.5),
        (0.05, 0.7, 0.3),
        (0.1, 0.45, 0.55),
        (0.2, 0.5, 0.5),
    ]

    # Act
    result = get_geometric_soft_labels(n, alphas)

    # Assert that unimodal smoothings for each class sum to one
    np.testing.assert_equal(np.mean(np.sum(result, axis=1)), 1.0)
    # Assert that true class has 1 - alpha probability mass
    np.testing.assert_array_equal(
        np.diagonal(result), 1.0 - np.array([alpha[0] for alpha in alphas])
    )


def test_get_geometric_soft_labels_asymmetric_c():
    # Arrange
    n = 5
    alphas = [
        (0.2, 0, 1),
        (0.1, 0.5, 0.5),
        (0.05, 0.7, 0.3),
        (0.1, 0.45, 0.55),
        (0.2, 1, 0),
    ]

    # Act
    result = get_geometric_soft_labels(n, alphas)

    # Assert that unimodal smoothings for each class sum to one
    np.testing.assert_equal(np.mean(np.sum(result, axis=1)), 1.0)
    # Assert that true class has 1 - alpha probability mass
    np.testing.assert_array_equal(
        np.diagonal(result), 1.0 - np.array([alpha[0] for alpha in alphas])
    )


def test_get_geometric_soft_labels_asymmetric_extreme_right():
    # Arrange
    n = 5
    alphas = [(0.2, 0, 1), (0.1, 0, 1), (0.05, 0, 1), (0.1, 0, 1), (0.2, 0, 1)]

    # Act
    result = get_geometric_soft_labels(n, alphas)

    # Assert that unimodal smoothings for each class sum to one
    np.testing.assert_equal(np.mean(np.sum(result, axis=1)), 1.0)


def test_get_geometric_soft_labels_asymmetric_extreme_left():
    # Arrange
    n = 5
    alphas = [(0.2, 1, 0), (0.1, 1, 0), (0.05, 1, 0), (0.1, 1, 0), (0.2, 1, 0)]

    # Act
    result = get_geometric_soft_labels(n, alphas)

    # Assert that unimodal smoothings for each class sum to one
    np.testing.assert_equal(np.mean(np.sum(result, axis=1)), 1.0)
