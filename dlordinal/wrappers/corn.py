import torch


class CORNClassifierWrapper:
    def __init__(self, classifier, **kwargs):
        # Check that the classifier implements predict and predict_proba
        if not all(
            hasattr(classifier, method) for method in ("predict", "predict_proba")
        ):
            raise ValueError(
                "The classifier must implement 'predict' and 'predict_proba' methods."
            )
        self.classifier = classifier

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped classifier."""
        return getattr(self.classifier, name)

    def predict(self, X):
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)

    def predict_proba(self, X):
        logits = self.classifier.predict_proba(X)

        # Convert logits to class labels using CORN logic
        probas = torch.sigmoid(logits)
        probas = torch.cumprod(probas, dim=1)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        num_classes = logits.shape[1] + 1
        probs = torch.zeros((len(predicted_labels), num_classes), dtype=torch.float32)
        probs[torch.arange(len(predicted_labels)), predicted_labels] = 1.0
        return probs.numpy()
