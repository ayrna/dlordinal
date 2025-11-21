import numpy as np
import pytest
import torch

from dlordinal.wrappers import CORNClassifierWrapper


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ]
)
def torch_device(request):
    return torch.device(request.param)


# --- Mock for the base classifier ---
class MockClassifier:
    """Mock classifier with required methods and an extra attribute for delegation."""

    def __init__(self, logits_output, device):
        # Mover los logits al dispositivo de prueba
        self._logits_output = logits_output.to(device)
        self.some_attribute = 42  # Attribute to test delegation

    def predict(self, X):
        # This method is not used by the wrapper, but is required
        raise NotImplementedError("Mock predict not implemented")

    def predict_proba(self, X):
        # Returns the CORN logits that the wrapper will process
        return self._logits_output


# --- Initialization Tests ---
# Estos tests ahora usan torch_device para inicializar el MockClassifier.
def test_initialization_success(torch_device):
    """Tests that initialization succeeds if the classifier has the required methods."""
    # Usamos un tensor vacío, pero lo movemos al dispositivo correcto.
    mock_classifier = MockClassifier(torch.empty(0), device=torch_device)
    wrapper = CORNClassifierWrapper(mock_classifier)
    assert wrapper.classifier == mock_classifier


def test_initialization_failure_missing_predict():
    """Tests that initialization fails if 'predict' is missing."""

    # Estos tests no crean MockClassifier, por lo que no necesitan torch_device.
    class BadClassifier:
        def predict_proba(self, X):
            return None

    with pytest.raises(
        ValueError,
        match="The classifier must implement 'predict' and 'predict_proba' methods.",
    ):
        CORNClassifierWrapper(BadClassifier())


def test_initialization_failure_missing_predict_proba():
    """Tests that initialization fails if 'predict_proba' is missing."""

    # Estos tests no crean MockClassifier, por lo que no necesitan torch_device.
    class BadClassifier:
        def predict(self, X):
            return None

    with pytest.raises(
        ValueError,
        match="The classifier must implement 'predict' and 'predict_proba' methods.",
    ):
        CORNClassifierWrapper(BadClassifier())


# --- Attribute Delegation Tests ---
def test_attribute_delegation(torch_device):
    """Tests that attributes are correctly delegated to the wrapped classifier."""
    # El mock es inicializado con el dispositivo de prueba.
    mock_classifier = MockClassifier(torch.empty(0), device=torch_device)
    wrapper = CORNClassifierWrapper(mock_classifier)

    # Check if an attribute that only exists in the mock is accessible via the wrapper
    assert wrapper.some_attribute == 42


# --- `predict_proba` Tests (CORN Logic) ---
@pytest.mark.parametrize(
    "input_logits, expected_probs",
    [
        # Case 1: All negative -> All classes 0 (Final Class = 0)
        # 3 logits = 4 classes (0, 1, 2, 3)
        (
            torch.tensor([[-5.0, -5.0, -5.0]], dtype=torch.float32),
            np.array([[1.0, 0.0, 0.0, 0.0]]),
        ),
        # Case 2: Logits resulting in P(y>=1)>0.5, P(y>=2)<=0.5 -> Class 1
        # Sigmoids: [~0.95, ~0.05, ~0.05]
        (
            torch.tensor([[3.0, -3.0, -3.0]], dtype=torch.float32),
            np.array([[0.0, 1.0, 0.0, 0.0]]),
        ),
        # Case 3: Logits resulting in P(y>=1)>0.5, P(y>=2)>0.5, P(y>=3)<=0.5 -> Class 2
        # Sigmoids: [~0.95, ~0.95, ~0.05]
        (
            torch.tensor([[3.0, 3.0, -3.0]], dtype=torch.float32),
            np.array([[0.0, 0.0, 1.0, 0.0]]),
        ),
        # Case 4: All positive -> P(y>=k) > 0.5 for all -> Class N
        (
            torch.tensor([[3.0, 3.0, 3.0]], dtype=torch.float32),
            np.array([[0.0, 0.0, 0.0, 1.0]]),
        ),
        # Case 5: Multiple samples (N=2), 5 logits = 6 classes (0 to 5)
        (
            torch.tensor(
                [[3.0, 3.0, 3.0, 3.0, -3.0], [-3.0, 3.0, 3.0, 3.0, 3.0]],
                dtype=torch.float32,
            ),
            np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        ),
    ],
)
def test_predict_proba_corn_logic(input_logits, expected_probs, torch_device):
    """Tests that predict_proba correctly applies the CORN logic on the specified device."""
    # El mock ahora mueve input_logits a torch_device
    mock_classifier = MockClassifier(input_logits, device=torch_device)
    wrapper = CORNClassifierWrapper(mock_classifier)

    # X is irrelevant for the mock
    X_dummy = None
    result_probs = wrapper.predict_proba(X_dummy)

    # La salida siempre es NumPy y está en la CPU, por lo que la comparación es segura
    assert np.array_equal(result_probs, expected_probs)
    assert result_probs.shape == expected_probs.shape
    assert result_probs.dtype == np.float32


# --- `predict` Tests ---
def test_predict_correct_argmax(torch_device):
    """Tests that predict returns the index with the highest probability on the specified device."""

    # Logits for a 4-class sample (0 to 3). Expected Class: 2
    input_logits = torch.tensor([[3.0, 3.0, -3.0]], dtype=torch.float32)

    # El mock mueve input_logits a torch_device
    mock_classifier = MockClassifier(input_logits, device=torch_device)
    wrapper = CORNClassifierWrapper(mock_classifier)

    X_dummy = None
    predicted_labels = wrapper.predict(X_dummy)

    # The output of predict must be a NumPy array
    assert isinstance(predicted_labels, np.ndarray)
    assert predicted_labels.dtype == np.int64  # argmax returns int64 by default

    # The predicted class is 2
    assert np.array_equal(predicted_labels, np.array([2]))
