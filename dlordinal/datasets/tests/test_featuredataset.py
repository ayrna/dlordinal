import numpy as np
import pandas as pd
import pytest
import torch

from dlordinal.datasets import FeatureDataset


@pytest.fixture
def sample_data(tmp_path):
    data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [2, 4, 6, 8, 10],
        "target": [0, 1, 0, 1, 0],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "sample_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.mark.no_gpu_ci
def test_feature_dataset_creation(sample_data):
    dataset = FeatureDataset(sample_data)
    assert isinstance(dataset, FeatureDataset)


@pytest.mark.no_gpu_ci
def test_feature_dataset(sample_data):
    dataset = FeatureDataset(sample_data)

    assert len(dataset) == 5

    # Check that the elements of X and targets are correct before normalising
    assert torch.allclose(
        dataset.X,
        torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0]]),
        atol=1e-4,
    )
    assert dataset.targets.dtype == torch.int64
    assert torch.equal(dataset.targets, torch.tensor([0, 1, 0, 1, 0]))
    assert dataset.classes.dtype == torch.int64
    assert torch.equal(dataset.classes, torch.tensor([0, 1]))

    dataset.normalize_X()

    # Check that X, X_mean and X_scale are correct after normalising X
    assert torch.allclose(
        dataset.X,
        torch.tensor(
            [
                [-1.4142, -1.4142],
                [-0.7071, -0.7071],
                [0.0, 0.0],
                [0.7071, 0.7071],
                [1.4142, 1.4142],
            ]
        ),
        atol=1e-4,
    )

    mean = torch.from_numpy(dataset.X_mean)
    scale = torch.from_numpy(dataset.X_scale)

    assert torch.allclose(mean, torch.tensor([3.0, 6.0], dtype=mean.dtype), atol=1e-4)
    assert torch.allclose(
        scale, torch.tensor([1.4142, 2.8284], dtype=scale.dtype), atol=1e-4
    )

    dataset.normalize_y()

    # Check that targets, y_mean and y_scale are correct after normalising y
    assert torch.allclose(
        dataset.targets,
        torch.tensor([[-0.8165], [1.2247], [-0.8165], [1.2247], [-0.8165]]),
        atol=1e-4,
    )

    mean = torch.from_numpy(dataset.y_mean)
    scale = torch.from_numpy(dataset.y_scale)

    assert torch.allclose(mean, torch.tensor(0.4000, dtype=mean.dtype), atol=1e-4)
    assert torch.allclose(scale, torch.tensor(0.4899, dtype=scale.dtype), atol=1e-4)

    # Check that the elements of X and targets are correct after normalising
    assert torch.allclose(
        dataset.X,
        torch.tensor(
            [
                [-1.4142, -1.4142],
                [-0.7071, -0.7071],
                [0.0, 0.0],
                [0.7071, 0.7071],
                [1.4142, 1.4142],
            ]
        ),
        atol=1e-4,
    )

    assert torch.allclose(
        dataset.targets,
        torch.tensor([[-0.8165], [1.2247], [-0.8165], [1.2247], [-0.8165]]),
        atol=1e-4,
    )


@pytest.fixture
def train_test_data(tmp_path):
    train = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 9.0], "y": [0, 1, 2, 1]}
    )
    test = pd.DataFrame({"a": [1.0, 5.0], "b": [3.0, 7.0], "y": [0, 2]})
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    return tmp_path / "train.csv", tmp_path / "test.csv"


@pytest.mark.no_gpu_ci
def test_integer_targets_work_with_classification_losses(train_test_data):
    dataset = FeatureDataset(train_test_data[0])
    _, y = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4)))
    assert y.dtype == torch.int64
    loss = torch.nn.CrossEntropyLoss()(torch.zeros(4, 3), y)
    assert torch.isfinite(loss)


@pytest.mark.no_gpu_ci
def test_continuous_targets_stay_float(tmp_path):
    csv_path = tmp_path / "continuous.csv"
    pd.DataFrame({"a": [1.0, 2.0, 3.0], "y": [0.5, 1.25, 2.0]}).to_csv(
        csv_path, index=False
    )
    dataset = FeatureDataset(csv_path)
    assert dataset.targets.dtype == torch.float32
    assert torch.allclose(dataset.targets, torch.tensor([0.5, 1.25, 2.0]))


@pytest.mark.no_gpu_ci
def test_normalize_with_train_statistics(train_test_data):
    train = FeatureDataset(train_test_data[0]).normalize_X().normalize_y()
    test = FeatureDataset(train_test_data[1])
    test.normalize_X(train.X_mean, train.X_scale)
    test.normalize_y(train.y_mean, train.y_scale)

    raw = pd.read_csv(train_test_data[1]).to_numpy()
    expected_X = (raw[:, :2] - train.X_mean) / train.X_scale
    expected_y = (raw[:, 2:] - train.y_mean) / train.y_scale
    assert torch.allclose(test.X, torch.tensor(expected_X, dtype=torch.float))
    assert torch.allclose(test.targets, torch.tensor(expected_y, dtype=torch.float))


@pytest.mark.no_gpu_ci
def test_normalize_with_scalar_statistics(train_test_data):
    dataset = FeatureDataset(train_test_data[1])
    dataset.normalize_y(0.5, 2.0)
    assert torch.allclose(dataset.targets, torch.tensor([[-0.25], [0.75]]))
    assert np.allclose(dataset.y_mean, [0.5])
    assert np.allclose(dataset.y_scale, [2.0])

    dataset.normalize_X(1.0, 2.0)
    assert torch.allclose(dataset.X, torch.tensor([[0.0, 1.0], [2.0, 3.0]]))
