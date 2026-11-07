import numpy as np
import torch


class CORNClassifierWrapper:
    """
    A wrapper that converts the output of a standard multi-output binary classifier
    into rank-consistent ordinal class predictions using the CORN (Classification
    and Ordinal Regression) transformation logic.

    This wrapper handles the final prediction steps:
    1. Applies the Sigmoid and Cumulative Product to the raw logits (N, J-1).
    2. ``predict`` determines the final ordinal class label (0 to J-1) based on a
    probability threshold (default 0.5).
    3. ``predict_proba`` returns the class probabilities (N, J) implied by the
    CORN logits.

    Parameters
    ----------
    classifier : object
        The base classifier instance (e.g., a PyTorch neural network module)
        that implements the `predict` and `predict_proba` methods. Its
        `predict_proba` must return raw logits of shape (N, J-1).
    threshold : float, optional
        The probability threshold (between 0 and 1) used to decide if the
        cumulative probability P(y >= k) corresponds to the next class level.
        The default is 0.5, based on standard binary classification practice,
        but can be tuned.

    Raises
    ------
    ValueError
        If the wrapped classifier does not implement both 'predict' and
        'predict_proba' methods.
    """

    def __init__(self, classifier, threshold=0.5):
        # Check that the classifier implements predict and predict_proba
        if not all(
            hasattr(classifier, method) for method in ("predict", "predict_proba")
        ):
            raise ValueError(
                "The classifier must implement 'predict' and 'predict_proba' methods."
            )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0.")

        self.classifier = classifier
        self.threshold = threshold

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped classifier."""
        # __getattr__ only runs when normal lookup fails. While the instance is
        # being unpickled or deep-copied, ``classifier`` is not set yet, and
        # delegating would call __getattr__("classifier") again forever. Dunder
        # lookups (e.g. __setstate__, __deepcopy__) are not delegated either.
        if name == "classifier" or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(self.classifier, name)

    def __call__(self, *args, **kwargs):
        """
        Delegates the call to the underlying classifier.
        This is necessary for the training loop in skorch/PyTorch
        which expects to call the module directly to get logits.
        """
        return self.classifier(*args, **kwargs)

    def predict(self, X):
        """
        Predicts the final ordinal class labels for the input data X.

        Parameters
        ----------
        X : array-like or torch.Tensor
            The input data to make predictions on.

        Returns
        -------
        numpy.ndarray
            An array of integer class labels (0 to J-1) of shape (N,).
        """
        # The predicted class is the number of cumulative probabilities
        # P(y > k) = prod_{i<=k} sigmoid(z_i) that exceed the threshold.
        cum_probas = self._cumulative_probas(X)
        predicted_labels = torch.sum(cum_probas > self.threshold, dim=1)
        return predicted_labels.cpu().numpy().astype(np.int64)

    def predict_proba(self, X):
        """
        Applies the CORN transformation to the raw logits to produce the class
        probabilities.

        The conditional probabilities :math:`\\sigma(z_k) = P(y > k \\mid y > k-1)`
        are chained into :math:`P(y > k) = \\prod_{i \\leq k} \\sigma(z_i)`, and the
        class probabilities are :math:`P(y = k) = P(y > k-1) - P(y > k)`, with
        :math:`P(y > -1) = 1` and :math:`P(y > J-1) = 0`. They are non-negative and
        sum to one, so they can be used with probabilistic metrics such as the
        ranked probability score. The class labels returned by :meth:`predict` use
        the threshold rule instead, so they are not always the argmax of these
        probabilities.

        Parameters
        ----------
        X : array-like or torch.Tensor
            The input data.

        Returns
        -------
        numpy.ndarray
            An array of class probabilities of shape (N, J), where J is the total
            number of classes.
        """
        cum_probas = self._cumulative_probas(X)
        n = cum_probas.shape[0]
        ones = torch.ones((n, 1), dtype=cum_probas.dtype, device=cum_probas.device)
        zeros = torch.zeros_like(ones)
        # P(y > k - 1) - P(y > k) for k = 0, ..., J-1
        probs = torch.cat([ones, cum_probas], dim=1) - torch.cat(
            [cum_probas, zeros], dim=1
        )
        return probs.clamp_min(0.0).cpu().numpy()

    def _cumulative_probas(self, X):
        """Returns P(y > k) for k = 0, ..., J-2 as a float32 tensor of shape (N, J-1)."""
        # Raw logits (N, J-1) from the wrapped classifier
        logits = self.classifier.predict_proba(X)
        logits = torch.as_tensor(logits, dtype=torch.float32)
        return torch.cumprod(torch.sigmoid(logits), dim=1)
