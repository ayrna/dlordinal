from sklearn.preprocessing import StandardScaler
import numpy as np

class StandardScaler3D(StandardScaler):
    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)

    def fit(self, X, y=None, sample_weight=None):
        X = X.reshape(X.shape[0], -1)
        return super().fit(X, y, sample_weight)

    def transform(self, X, copy=None):
        initial_shape = X.shape
        X = X.reshape(X.shape[0], -1)
        transformed_X = super().transform(X, copy)
        return np.reshape(transformed_X, newshape=initial_shape)