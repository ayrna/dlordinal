import numpy as np
import torch

from dlordinal.output_layers.clm_loss_at import CLMAT


def _test_probas(clm):
    projections = torch.rand(32, 1)
    probas, _, _, _ = clm(projections)
    total_probas = torch.sum(probas, dim=1)
    assert torch.allclose(total_probas, torch.ones_like(total_probas))
    assert isinstance(probas, torch.Tensor)

    return projections, probas, total_probas


def test_clm_creation():
    num_classes = 3
    link_function = "logit"
    min_distance = 0.0

    clm = CLMAT(
        num_classes=num_classes, link_function=link_function, min_distance=min_distance
    )

    assert isinstance(clm, CLMAT)


def test_clm_probas():
    num_classes = 5
    link_function = "logit"
    min_distance = 0.0

    clm = CLMAT(
        num_classes=num_classes, link_function=link_function, min_distance=min_distance
    )

    _test_probas(clm)


def test_clm_thresholds():
    num_classes = 5
    link_function = "logit"
    min_distance = 0.0

    clm = CLMAT(
        num_classes=num_classes,
        use_gammas=True,
        link_function=link_function,
        min_distance=min_distance,
    )

    thresholds = clm._convert_thresholds(
        clm.first_threshold_, clm.thresholds_gammas_, min_distance
    )
    expected_thresholds = torch.tensor([float(i) for i in range(num_classes - 2 + 1)])

    assert (
        thresholds.shape[0]
        == clm.thresholds_gammas_.shape[0] + clm.first_threshold_.shape[0]
    )

    assert torch.allclose(thresholds, expected_thresholds)

    _test_probas(clm)


def test_clm_link_functions():
    for link in ["logit", "probit", "cloglog"]:
        for num_classes in range(3, 12):
            clm = CLMAT(num_classes=num_classes, link_function=link, min_distance=0.0)
            assert clm.link_function == link
            assert clm.num_classes == num_classes

            _test_probas(clm)


def test_clm_all_combinations():
    for link in ["logit", "probit", "cloglog"]:
        for num_classes in range(3, 12):
            for min_distance in np.linspace(0.0, 0.1, 10):
                clm = CLMAT(
                    num_classes=num_classes,
                    link_function=link,
                    min_distance=min_distance,
                )
                assert clm.link_function == link
                assert clm.num_classes == num_classes
                assert clm.min_distance == min_distance

                _test_probas(clm)


def test_clm_thresholds_exhaustive():
    num_classes = 5
    link_function = "logit"
    min_distance = 0.0

    clm = CLMAT(
        num_classes=num_classes,
        use_gammas=True,
        link_function=link_function,
        min_distance=min_distance,
    )

    gammas_1 = torch.Tensor([0.2, 0.05, 0.7])
    th_base_1 = torch.Tensor([0.5])
    th_1 = clm._convert_thresholds(th_base_1, gammas_1, min_distance)
    expected_th_1 = torch.Tensor([0.5, 0.54, 0.5425, 1.0325])
    assert torch.allclose(th_1, expected_th_1)

    gammas_2 = torch.Tensor([7.2, 0.4, 4.3])
    th_base_2 = torch.Tensor([1.2])
    th_2 = clm._convert_thresholds(th_base_2, gammas_2, min_distance)
    expected_th_2 = torch.Tensor([1.2, 53.04, 53.2, 71.69])
    assert torch.allclose(th_2, expected_th_2)

    gammas_3 = torch.Tensor([-5.6, 6.1, -8.9])
    th_base_3 = torch.Tensor([-0.7])
    th_3 = clm._convert_thresholds(th_base_3, gammas_3, min_distance)
    expected_th_3 = torch.Tensor([-0.7, 30.66, 67.87, 147.08])
    assert torch.allclose(th_3, expected_th_3)


def test_clm_probas_from_projection_and_thresholds():
    num_classes = 5
    link_function = "logit"
    min_distance = 0.0

    clm = CLMAT(
        num_classes=num_classes, link_function=link_function, min_distance=min_distance
    )

    th = torch.Tensor([0, 1, 2, 3])
    wx = torch.Tensor([2.2])
    probas, _, _, _ = clm._clm(wx, th)
    expected_probas = torch.Tensor(
        [
            0.0997504891196851,
            0.131724727381297,
            0.21869078618654,
            0.23980847844009,
            0.310025518872387,
        ]
    )
    assert torch.allclose(probas, expected_probas, atol=1e-06)

    th = torch.Tensor([0.5, 0.54, 0.5425, 1.0325])
    wx = torch.Tensor([0.25])
    probas, _, _, _ = clm._clm(wx, th)
    expected_probas = torch.Tensor(
        [
            0.562176500885798,
            0.00981963204572056,
            0.0006119309317415,
            0.113610607034421,
            0.313781329102319,
        ]
    )
    assert torch.allclose(probas, expected_probas, atol=1e-06)

    th = torch.Tensor([1.2, 53.04, 53.2, 71.69])
    wx = torch.Tensor([-10.0])
    probas, _, _, _ = clm._clm(wx, th)
    expected_probas = torch.Tensor(
        [0.99998633, 0.00001367, 0.00000000, 0.00000000, 0.00000000]
    )
    assert torch.allclose(probas, expected_probas, atol=1e-06)

    th = torch.Tensor([-0.7, 30.66, 67.87, 147.08])
    wx = torch.Tensor([10.0])
    probas, _, _, _ = clm._clm(wx, th)
    expected_probas = torch.Tensor(
        [0.00002254, 0.99997745, 0.00000000, 0.00000000, 0.00000000]
    )
    assert torch.allclose(probas, expected_probas, atol=1e-06)


def test_not_using_gammas_forward():
    num_classes = 5
    link_function = "logit"
    min_distance = 0.0
    use_gammas = False

    clm = CLMAT(
        num_classes=num_classes,
        use_gammas=use_gammas,
        link_function=link_function,
        min_distance=min_distance,
    )

    clm.thresholds_ = torch.nn.Parameter(
        data=torch.Tensor([1.2, 53.04, 53.2, 71.69]),
        requires_grad=True,
    )
    wx = torch.Tensor([-10.0])
    probas, _, _, _ = clm.forward(wx)
    expected_probas = torch.Tensor(
        [0.99998633, 0.00001367, 0.00000000, 0.00000000, 0.00000000]
    )
    assert torch.allclose(probas, expected_probas, atol=1e-06)

    clm.thresholds_ = torch.nn.Parameter(
        data=torch.Tensor([-0.43, 2.770521, 6.810621, 7.771021]),
        requires_grad=True,
    )
    wx = torch.Tensor([2.986])
    probas, _, _, _ = clm.forward(wx)
    expected_probas = torch.Tensor(
        [0.03179915, 0.41453857, 0.53230180, 0.01307574, 0.00828474]
    )
    assert torch.allclose(probas, expected_probas, atol=1e-06)


def test_not_using_weights_forward():
    clm = CLMAT(
        num_classes=5,
        use_weights=True,
        use_gammas=False,
        link_function="logit",
        min_distance=0.0,
    )

    clm.thresholds_ = torch.nn.Parameter(
        data=torch.Tensor([-0.43, 2.770521, 6.810621, 7.771021]),
        requires_grad=True,
    )
    clm.weights_ = torch.nn.Parameter(
        data=torch.Tensor([1.0]),
        requires_grad=True,
    )
    x = torch.Tensor([2.986])
    probas, _, _, _ = clm.forward(x)
    expected_probas = torch.Tensor(
        [0.03179915, 0.41453857, 0.53230180, 0.01307574, 0.00828474]
    )
    assert torch.allclose(probas, expected_probas, atol=1e-06)

    clm = CLMAT(
        num_classes=5,
        use_weights=True,
        use_gammas=False,
        link_function="logit",
        min_distance=0.0,
    )

    clm.thresholds_ = torch.nn.Parameter(
        data=torch.Tensor([-2.0982, -2.04320975, -0.23418475, 1.3447582836]),
        requires_grad=True,
    )
    clm.weights_ = torch.nn.Parameter(
        data=torch.Tensor([0.879]),
        requires_grad=True,
    )
    x = torch.Tensor([1.1345])
    probas, _, _, _ = clm.forward(x)
    expected_probas = torch.Tensor(
        [0.04329634, 0.00233587, 0.18030248, 0.36008446, 0.41398084]
    )
    assert torch.allclose(probas, expected_probas, atol=1e-06)


def test_clm_output_to_loss():
    clm = CLMAT(
        num_classes=5,
        use_weights=True,
        use_gammas=False,
        link_function="logit",
        min_distance=0.0,
    )

    clm.thresholds_ = torch.nn.Parameter(
        data=torch.Tensor([-0.43, 2.770521, 6.810621, 7.771021]),
        requires_grad=True,
    )
    clm.weights_ = torch.nn.Parameter(
        data=torch.Tensor([1.0]),
        requires_grad=True,
    )
    x = torch.Tensor([2.986])
    probas, w, x, th = clm.forward(x)
    expected_probas = torch.Tensor(
        [0.03179915, 0.41453857, 0.53230180, 0.01307574, 0.00828474]
    )
    assert torch.allclose(probas, expected_probas, atol=1e-06)
    assert w == torch.Tensor([1.0])
    assert x == torch.Tensor([2.986])
    assert torch.allclose(th, torch.Tensor([-0.43, 2.770521, 6.810621, 7.771021]))
