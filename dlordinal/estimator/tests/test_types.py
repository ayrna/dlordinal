import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dlordinal.estimator import PytorchEstimator


@pytest.fixture
def setup_estimator():
    # Model
    model = torch.nn.Linear(10, 5)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")
    max_iter = 5

    estimator = PytorchEstimator(model, loss_fn, optimizer, device, max_iter)

    return estimator


class DummyDataLoader(DataLoader):
    def __init__(self):
        super().__init__(dataset=None)


def create_example_dataloader(batch_size, num_samples, input_size):
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, 2, (num_samples,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def test_fit_X_dataloader_y_none(setup_estimator):
    dummy_dataloader = DummyDataLoader()

    y = torch.tensor([1, 0, 1, 0])

    with pytest.raises(ValueError, match="If X is a DataLoader, y must be None"):
        setup_estimator.fit(X=dummy_dataloader, y=y)


def test_fit_X_tensor_y_none(setup_estimator):
    X = torch.tensor([1, 0, 1, 0])

    with pytest.raises(ValueError, match="If X is a torch Tensor, y must not be None"):
        setup_estimator.fit(X=X, y=None)


def test_fit_X_tensor_y_not_tensor(setup_estimator):
    X = torch.tensor([1, 0, 1, 0])

    with pytest.raises(ValueError, match="y must be a torch.Tensor"):
        setup_estimator.fit(X=X, y=1)


def test_fit_X_not_dataloader_or_tensor(setup_estimator):
    X = 1

    with pytest.raises(ValueError, match="X must be a DataLoader or a torch Tensor"):
        setup_estimator.fit(X=X, y=None)


def test_predict_proba_X_none(setup_estimator):
    train_dataloader = create_example_dataloader(16, num_samples=100, input_size=10)

    estimator = setup_estimator.fit(X=train_dataloader, y=None)

    with pytest.raises(ValueError, match="X must be a DataLoader or a torch Tensor"):
        estimator.predict_proba(X=None)


def test_predict_proba_X_not_dataloader_or_tensor(setup_estimator):
    train_dataloader = create_example_dataloader(16, num_samples=100, input_size=10)

    estimator = setup_estimator.fit(X=train_dataloader, y=None)

    with pytest.raises(ValueError, match="X must be a DataLoader or a torch Tensor"):
        estimator.predict_proba(X=1)
