from typing import Optional, Union

import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader


class PytorchEstimator(BaseEstimator):
    """
    Wrapper around a Pytorch ``nn.Module`` implementing
    the default estimator interface defined by ``scikit-learn``.

    Parameters
    ----------
    model : torch.nn.Module
        A Pytorch model.
    loss_fn : torch.nn.Module
        A Pytorch loss function.
    optimizer : torch.optim.Optimizer
        A Pytorch optimizer.
    device : torch.device
        A Pytorch device.
    max_iter : int
        The maximum number of iterations.
    \**kwargs : dict
        Additional keyword arguments.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        max_iter: int,
        **kwargs,
    ):
        self.kwargs = kwargs
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.max_iter = max_iter

    def fit(
        self,
        X: Union[DataLoader, torch.Tensor],
        y: Optional[Union[torch.Tensor, None]] = None,
    ):
        """
        fit() is a method that fits the model to the training data.

        Parameters
        ----------
        X : Union[DataLoader, torch.Tensor]
            The training data.
        y : Optional[Union[torch.Tensor, None]], default=None
            The training labels, only used if X is a ``torch.Tensor``.
        """

        # Check if X is a DataLoader
        if isinstance(X, DataLoader):
            if y is None:
                print("Training ...")
                self.model.train()

                # Iterate over epochs
                for epoch in range(self.max_iter):
                    print(f"Epoch {epoch+1}/{self.max_iter}")

                    # Iterate over batches
                    for _, (X_batch, y_batch) in enumerate(X):
                        self._fit(X_batch, y_batch)

            else:
                raise ValueError("If X is a DataLoader, y must be None")

        # Check if X is a torch Tensor
        elif isinstance(X, torch.Tensor):
            if y is None:
                raise ValueError("If X is a torch Tensor, y must not be None")

            # Check if y is a torch Tensor
            elif isinstance(y, torch.Tensor):
                print("Training ...")
                self.model.train()

                # Iterate over epochs
                for epoch in range(self.max_iter):
                    print(f"Epoch {epoch+1}/{self.max_iter}")
                    self._fit(X, y)

            else:
                raise ValueError("y must be a torch.Tensor")

        else:
            raise ValueError("X must be a DataLoader or a torch Tensor")

        return self

    def _fit(self, X, y):
        """
        _fit() is a private method that performs a forward pass, computes the loss
        and performs backpropagation.

        Parameters
        ----------
        X : torch.Tensor
            The training data.
        y : torch.Tensor
            The training labels.
        """
        X, y = X.to(self.device), y.to(self.device)

        # Forward pass
        pred = self.model(X)
        loss = self.loss_fn(pred, y)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict_proba(self, X: Union[DataLoader, torch.Tensor]):
        """
        predict_proba() is a method that predicts the probability of each class.

        Parameters
        ----------
        X : Union[DataLoader, torch.Tensor]
            The data to predict.
        """
        if X is None:
            raise ValueError("X must be a DataLoader or a torch Tensor")

        else:
            # check if X is a DataLoader
            if isinstance(X, DataLoader):
                print("Predicting ...")
                self.model.eval()
                predictions = []

                # Iterate over batches
                for _, (X_batch, _) in enumerate(X):
                    predictions_batch = self._predict_proba(X_batch)
                    predictions.append(predictions_batch)

                # Concatenate predictions
                predictions = torch.cat(predictions)
                return predictions

            # check if X is a torch Tensor
            elif isinstance(X, torch.Tensor):
                print("Predicting ...")
                self.model.eval()
                return self._predict_proba(X)

            else:
                raise ValueError("X must be a DataLoader or a torch Tensor")

    def _predict_proba(self, X):
        """
        _predict_proba() is a private method that predicts the probability
        of each class.

        Parameters
        ----------
        X : torch.Tensor
            The data to predict.
        """
        X = X.to(self.device)
        pred = self.model(X)
        probabilities = F.softmax(pred, dim=1)
        return probabilities

    def predict(self, X: Union[DataLoader, torch.Tensor]):
        """
        predict() is a method that predicts the class of each sample.

        Parameters
        ----------
        X : Union[DataLoader, torch.Tensor]
            The data to predict.
        """
        pred = self.predict_proba(X)
        return torch.argmax(pred, dim=1)
