from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """Dataset torch implementation for a standard dataset that contains several features
    that are organised in a tabular way in a csv file. The last column is the target
    variable.

    Example
    -------
    >>> train_data = FeatureDataset("train.csv")
    >>> train_data.normalize_X()
    >>> train_data.normalize_y()
    >>> train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    >>> for X, y in train_loader:
    >>>     print(X.shape, y.shape)
    >>> test_data = FeatureDataset("test.csv")
    >>> test_data.normalize_X(train_data.X_mean, train_data.X_scale)
    >>> test_data.normalize_y(train_data.y_mean, train_data.y_scale)
    >>> test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    >>> for X, y in test_loader:
    >>>     print(X.shape, y.shape)
    """

    def __init__(self, filename):
        """
        Parameters
        ----------
        filename : str
            Path to the csv file containing the dataset.
        """

        df = pd.read_csv(filename)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        self.X_mean: float = 0.0
        self.X_scale: float = 0.0
        self.y_mean: float = 0.0
        self.y_scale: float = 0.0

        self.X = torch.tensor(X, dtype=torch.float)
        self.targets = torch.tensor(y, dtype=torch.float)
        self.classes = torch.tensor(np.unique(y), dtype=torch.float)

    def get_valid_shape_array(self, v: ArrayLike):
        """Convert the input ArrayLike object to a 2D numpy array with shape (n, 1)
        if it is a 1D array.

        Parameters
        ----------
        v : ArrayLike
            Input array.

        Returns
        -------
        v : np.ndarray
            2D numpy array with shape (n, 1).
        """

        if isinstance(v, pd.Series):
            v = v.values  # type: ignore
        if len(v.shape) == 1:  # type: ignore
            v = np.reshape(v, (-1, 1))
        return v

    def _get_normalized(
        self,
        v: np.ndarray,
        mean: Optional[ArrayLike] = None,
        scale: Optional[ArrayLike] = None,
    ) -> Tuple[np.ndarray, ArrayLike, ArrayLike]:
        sc = StandardScaler()
        if mean is not None and scale is not None:
            sc.mean_ = mean
            sc.scale_ = scale
        else:
            sc.fit(v)
        return sc.transform(v), sc.mean_, sc.scale_

    def normalize_X(
        self, mean: Optional[ArrayLike] = None, scale: Optional[ArrayLike] = None
    ):
        """Standardize the features of the dataset.
        If mean and scale are not provided, they are computed from the dataset.
        If they are provided, they are used to standardize the dataset.

        Parameters
        ----------
        mean : array-like, default=None
            Mean of the dataset.
        scale : array-like, default=None
            Scale of the dataset.

        Returns
        -------
        self: FeatureDataset
            The dataset with standardized features.

        Example
        -------
        >>> train_data = FeatureDataset("train.csv")
        >>> train_data.normalize_X()
        >>> test_data = FeatureDataset("test.csv")
        >>> test_data.normalize_X(train_data.X_mean, train_data.X_scale)
        """

        self.X, self.X_mean, self.X_scale = self._get_normalized(
            self.get_valid_shape_array(self.X), mean, scale
        )
        self.X = torch.tensor(self.X, dtype=torch.float)
        return self

    def normalize_y(self, mean: ArrayLike = None, scale: ArrayLike = None):
        """Standardize the target variable of the dataset.
        If mean and scale are not provided, they are computed from the dataset.
        If they are provided, they are used to standardize the dataset.

        Parameters
        ----------
        mean : array-like, default=None
            Mean of the dataset.
        scale : array-like, default=None
            Scale of the dataset.

        Returns
        -------
        self: FeatureDataset
            The dataset with standardized target variable.

        Example
        -------
        >>> train_data = FeatureDataset("train.csv")
        >>> train_data.normalize_y()
        >>> test_data = FeatureDataset("test.csv")
        >>> test_data.normalize_y(train_data.y_mean, train_data.y_scale)
        """

        self.targets, self.y_mean, self.y_scale = self._get_normalized(
            self.get_valid_shape_array(self.targets), mean, scale
        )
        self.targets = torch.tensor(self.targets, dtype=torch.float)
        return self

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.X[idx], self.targets[idx]
