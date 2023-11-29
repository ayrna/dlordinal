import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from ..featuredataset import FeatureDataset


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


def test_feature_dataset_creation(sample_data):
    dataset = FeatureDataset(sample_data)
    assert isinstance(dataset, FeatureDataset)


def test_feature_dataset(sample_data):
    dataset = FeatureDataset(sample_data)

    assert len(dataset) == 5

    # Check that the elements of X and targets are correct before normalising
    assert torch.allclose(
        dataset.X,
        torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0]]),
        atol=1e-4,
    )
    assert torch.allclose(
        dataset.targets, torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]), atol=1e-4
    )

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
