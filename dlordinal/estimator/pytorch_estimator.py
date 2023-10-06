from typing import Optional, Union

import numpy as np
import torch
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader

"""
    PytorchEstimator is a class that implements a Pytorch estimator.
"""


class PytorchEstimator(BaseEstimator):
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

    """
        fit() is a method that trains the model.
    """

    def fit(
        self,
        X: Union[DataLoader, torch.Tensor],
        y: Optional[Union[torch.Tensor, None]] = None,
    ):
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

    """
        _fit() is a private method that performs a forward pass, computes the loss 
        and performs backpropagation.
    """

    def _fit(self, X, y):
        X, y = X.to(self.device), y.to(self.device)

        # Forward pass
        pred = self.model(X)
        loss = self.loss_fn(pred, y)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    """
        predict_proba() is a method that predicts the probability of each class.
    """

    def predict_proba(self, X: Union[DataLoader, torch.Tensor]):
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
                return self._predict(X)

            else:
                raise ValueError("X must be a DataLoader or a torch Tensor")

    """
        _predict_proba() is a private method that predicts the probability
        of each class.
    """

    def _predict_proba(self, X):
        X = X.to(self.device)
        pred = self.model(X)
        return pred

    """
        predict() is a method that predicts the class of each sample.
    """

    def predict(self, X: Union[DataLoader, torch.Tensor]):
        pred = self.predict_proba(X)
        return torch.argmax(pred, dim=1)
