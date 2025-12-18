import torch


class CORNClassifierWrapper:
    """
    A wrapper that converts the output of a standard multi-output binary classifier
    into rank-consistent ordinal class predictions using the CORN (Classification
    and Ordinal Regression) transformation logic.

    This wrapper handles the final prediction steps:
    1. Applies the Sigmoid and Cumulative Product to the raw logits (N, J-1).
    2. Determines the final ordinal class label (0 to J-1) based on a probability
    threshold (default 0.5).

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
        # Note: The output of predict_proba is already one-hot, so argmax is simple
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)

    def predict_proba(self, X):
        """
        Applies the CORN transformation logic to the raw logits to produce
        one-hot encoded ordinal class probabilities.

        The ordinal class is determined by counting the number of cumulative
        probabilities P(y >= k) that exceed the set threshold.

        Parameters
        ----------
        X : array-like or torch.Tensor
            The input data.

        Returns
        -------
        numpy.ndarray
            An array of one-hot encoded probabilities of shape (N, J), where
            J is the total number of classes.
        """
        # Step 1: Get raw logits (N, J-1) from the wrapped classifier
        logits = self.classifier.predict_proba(X)

        # Ensure logits are torch.Tensor and on the correct device for processing
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits, dtype=torch.float32)

        device = logits.device

        # Step 2: Apply Sigmoid (N, J-1) -> P(y >= k)
        probas = torch.sigmoid(logits)

        # Step 3: Apply Cumulative Product. (Used for stability/consistency, though
        # the thresholding decision is often based on the raw sigmoid output).
        probas = torch.cumprod(probas, dim=1)

        # Step 4: Determine the predicted level based on the variable threshold
        # The predicted class 'c' is the count of how many P(y >= k) probabilities
        # are greater than self.threshold.
        predict_levels = probas > self.threshold
        predicted_labels = torch.sum(predict_levels, dim=1)  # Shape (N,)

        # Step 5: Convert the predicted integer labels to one-hot probabilities
        num_classes = logits.shape[1] + 1

        # Initialize an (N, J) probability tensor
        probs = torch.zeros(
            (len(predicted_labels), num_classes), dtype=torch.float32, device=device
        )

        # Set the probability of the predicted class to 1.0 (one-hot)
        probs[torch.arange(len(predicted_labels), device=device), predicted_labels] = (
            1.0
        )

        return probs.cpu().numpy()
