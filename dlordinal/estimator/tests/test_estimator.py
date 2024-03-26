import numpy as np
import torch
from torch import cuda
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

from dlordinal.estimator import PytorchEstimator


def test_pytorch_estimator_creation():
    # Model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 6)

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = "cuda" if cuda.is_available() else "cpu"
    model = model.to(device)

    max_iter = 5

    estimator = PytorchEstimator(model, loss_fn, optimizer, device, max_iter)

    assert isinstance(estimator, PytorchEstimator)


def create_example_dataloader(batch_size, num_samples, input_size):
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, 2, (num_samples,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def calculate_loss(model, loss_fn, dataloader):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in dataloader:
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()

    return total_loss / len(dataloader.dataset)


def test_pytorch_estimator_fit_Dataloader():
    input_size = 10
    num_classes = 3
    batch_size = 16

    # Create an example DataLoader for training
    train_dataloader = create_example_dataloader(
        batch_size, num_samples=100, input_size=input_size
    )

    # Create an example DataLoader for prediction
    test_dataloader = create_example_dataloader(
        batch_size, num_samples=50, input_size=input_size
    )

    model = torch.nn.Linear(input_size, num_classes)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")
    max_iter = 5

    estimator = PytorchEstimator(model, loss_fn, optimizer, device, max_iter)

    # Verifies the training flow
    # initial_loss = calculate_loss(model, loss_fn, test_dataloader)
    estimator.fit(train_dataloader)
    final_loss = calculate_loss(model, loss_fn, test_dataloader)

    assert not np.isnan(final_loss)
    assert not np.isinf(final_loss)


def test_pytorch_estimator_fit_Tensor():
    input_size = 10
    num_classes = 3

    # Create an example of training data
    X = torch.randn(100, input_size)

    # Create an example of training labels
    y = torch.randint(0, num_classes, (100,))
    print("shape of y")
    print(y.shape)

    model = torch.nn.Linear(input_size, num_classes)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")
    max_iter = 5

    estimator = PytorchEstimator(model, loss_fn, optimizer, device, max_iter)

    # Verifies the training flow
    estimator.fit(X=X, y=y)


def test_pytorch_estimator_predict():
    input_size = 10
    num_classes = 3
    batch_size = 16

    # Create an example DataLoader for training
    train_dataloader = create_example_dataloader(
        batch_size, num_samples=100, input_size=input_size
    )

    # Create an example DataLoader for prediction
    test_dataloader = create_example_dataloader(
        batch_size, num_samples=50, input_size=input_size
    )

    model = torch.nn.Linear(input_size, num_classes)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")
    max_iter = 5

    estimator = PytorchEstimator(model, loss_fn, optimizer, device, max_iter)

    # Verifies the training flow
    estimator.fit(train_dataloader)

    # Verifies the prediction flow
    predictions = estimator.predict(test_dataloader)

    # Check that the predictions have the correct size
    assert predictions.shape == (50,)
    assert len(predictions) == 50

    # Check that the predictions are values in the range [0, num_classes)
    assert np.all(predictions >= 0) and np.all(predictions < num_classes)


def test_pytorch_estimator_predict_proba_dataloader():
    input_size = 10
    num_classes = 5
    batch_size = 16

    # Create an example DataLoader for training
    train_dataloader = create_example_dataloader(
        batch_size, num_samples=100, input_size=input_size
    )

    # Create an example DataLoader for prediction
    test_dataloader = create_example_dataloader(
        batch_size, num_samples=50, input_size=input_size
    )

    model = torch.nn.Linear(input_size, num_classes)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")
    max_iter = 5

    estimator = PytorchEstimator(model, loss_fn, optimizer, device, max_iter)

    # Verifies the training flow
    estimator.fit(train_dataloader)

    # Verifies the prediction flow
    probabilities = estimator.predict_proba(test_dataloader)

    # Check that the probabilities have the correct shape
    assert probabilities.shape == (50, 5)

    # Verify that the sum of the probabilities for each example is close to 1.
    assert np.allclose(np.sum(probabilities, axis=1), np.ones(50), atol=1e-5)

    # Verify that the probabilities are in the range [0, 1]
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)


def test_pytorch_estimator_predict_proba_tensor():
    input_size = 10
    num_classes = 3

    # Create an example of training data
    X = torch.randn(100, input_size)

    # Create an example of training labels
    y = torch.randint(0, num_classes, (100,))
    print("shape of y")
    print(y.shape)

    model = torch.nn.Linear(input_size, num_classes)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")
    max_iter = 5

    estimator = PytorchEstimator(model, loss_fn, optimizer, device, max_iter)

    # Verifies the training flow
    estimator.fit(X=X, y=y)

    # minimum, maximum, (number of samples, number of features)
    y_test = torch.randint(0, num_classes, (50, 10))
    y_test = y_test.float()

    probabilities = estimator.predict_proba(y_test)

    # Check that the probabilities have the correct shape
    assert probabilities.shape == (50, 3)

    # Check that the sum of the probabilities for each example is close to 1.
    assert np.allclose(np.sum(probabilities, axis=1), np.ones(50), atol=1e-5)

    # Check that the probabilities are in the range [0, 1]
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
