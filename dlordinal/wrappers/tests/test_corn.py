import numpy as np
import pytest
import torch

# Assuming CORNClassifierWrapper is available
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
    """Fixture to parametrize tests across 'cpu' and 'cuda' (if available)."""
    return torch.device(request.param)


# --- Mock for the base classifier ---
class MockClassifier:
    """Mock classifier with required methods and an extra attribute for delegation."""

    def __init__(self, logits_output, device):
        # Move logits to the test device
        self._logits_output = logits_output.to(device)
        self.some_attribute = 42  # Attribute to test delegation

    def predict(self, X):
        # This method is not used by the wrapper, but is required
        raise NotImplementedError("Mock predict not implemented")

    def predict_proba(self, X):
        # Returns the CORN logits that the wrapper will process
        return self._logits_output


# --- Initialization Tests ---


def test_initialization_success(torch_device):
    """Tests that initialization succeeds and the default threshold is 0.5."""
    mock_classifier = MockClassifier(torch.empty(0), device=torch_device)
    wrapper = CORNClassifierWrapper(mock_classifier)
    assert wrapper.classifier == mock_classifier
    # Check default threshold
    assert wrapper.threshold == 0.5


def test_initialization_with_custom_threshold(torch_device):
    """Tests that initialization succeeds with a custom threshold."""
    mock_classifier = MockClassifier(torch.empty(0), device=torch_device)
    custom_threshold = 0.8
    wrapper = CORNClassifierWrapper(mock_classifier, threshold=custom_threshold)
    assert wrapper.threshold == custom_threshold


def test_initialization_failure_missing_predict():
    """Tests that initialization fails if 'predict' is missing."""

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

    class BadClassifier:
        def predict(self, X):
            return None

    with pytest.raises(
        ValueError,
        match="The classifier must implement 'predict' and 'predict_proba' methods.",
    ):
        CORNClassifierWrapper(BadClassifier())


def test_initialization_failure_invalid_threshold():
    """Tests that initialization fails if the threshold is outside [0, 1]."""
    mock_classifier = MockClassifier(torch.empty(0), device="cpu")
    with pytest.raises(
        ValueError,
        match="Threshold must be between 0.0 and 1.0.",
    ):
        CORNClassifierWrapper(mock_classifier, threshold=1.1)


# --- Attribute Delegation Tests ---
def test_attribute_delegation(torch_device):
    """Tests that attributes are correctly delegated to the wrapped classifier."""
    mock_classifier = MockClassifier(torch.empty(0), device=torch_device)
    wrapper = CORNClassifierWrapper(mock_classifier)

    # Check if an attribute that only exists in the mock is accessible via the wrapper
    assert wrapper.some_attribute == 42


# --- `predict_proba` Tests (CORN Logic with Threshold) ---
# Parameters for logic tests: (input_logits, threshold, expected_probs)
@pytest.mark.parametrize(
    "input_logits, threshold, expected_probs",
    [
        # Case 1 (Default Threshold 0.5): All negative -> Class 0
        (
            torch.tensor([[-5.0, -5.0, -5.0]], dtype=torch.float32),
            0.5,
            np.array([[1.0, 0.0, 0.0, 0.0]]),
        ),
        # Case 2 (Default Threshold 0.5): Logits -> Class 1
        # Sigmoids: [~0.95, ~0.05, ~0.05] -> CumProd P(y>=k) > 0.5: [T, F, F] -> Sum=1
        (
            torch.tensor([[3.0, -3.0, -3.0]], dtype=torch.float32),
            0.5,
            np.array([[0.0, 1.0, 0.0, 0.0]]),
        ),
        # Case 3 (Custom Threshold 0.9): Same logits as Case 2.
        # P(y>=1) ~ 0.95. Since 0.95 > 0.9, the result is still Class 1.
        (
            torch.tensor([[3.0, -3.0, -3.0]], dtype=torch.float32),
            0.9,
            np.array([[0.0, 1.0, 0.0, 0.0]]),
        ),
        # Case 4 (Custom Threshold 0.99): Same logits as Case 2.
        # P(y>=1) ~ 0.95. Since 0.95 < 0.99, P(y>=1) fails. Result is Class 0.
        (
            torch.tensor([[3.0, -3.0, -3.0]], dtype=torch.float32),
            0.99,
            np.array([[1.0, 0.0, 0.0, 0.0]]),
        ),
        # Case 5 (Default Threshold 0.5): Logits -> Class 2
        # Sigmoids: [~0.95, ~0.95, ~0.05] -> CumProd P(y>=k) > 0.5: [T, T, F] -> Sum=2
        (
            torch.tensor([[3.0, 3.0, -3.0]], dtype=torch.float32),
            0.5,
            np.array([[0.0, 0.0, 1.0, 0.0]]),
        ),
        # Case 6 (Custom Threshold 0.9): Same logits as Case 5.
        # CumProd P(y>=1)~0.95 > 0.9 (T). CumProd P(y>=2)~0.9025 > 0.9 (T).
        # Result is Class 2.
        (
            torch.tensor([[3.0, 3.0, -3.0]], dtype=torch.float32),
            0.9,
            np.array([[0.0, 0.0, 1.0, 0.0]]),
        ),
        # Case 7 (Custom Threshold 0.92): Same logits as Case 5.
        # CumProd P(y>=1)~0.95 (T). CumProd P(y>=2)~0.9025 < 0.92 (F).
        # Result is Class 1.
        (
            torch.tensor([[3.0, 3.0, -3.0]], dtype=torch.float32),
            0.92,
            np.array([[0.0, 1.0, 0.0, 0.0]]),
        ),
        # Case 8 (Default 0.5): All positive -> Class N (Class 3)
        (
            torch.tensor([[3.0, 3.0, 3.0]], dtype=torch.float32),
            0.5,
            np.array([[0.0, 0.0, 0.0, 1.0]]),
        ),
    ],
)
def test_predict_proba_corn_logic(
    input_logits, threshold, expected_probs, torch_device
):
    """Tests that predict_proba correctly applies the CORN logic using
    a variable threshold on the specified device."""

    # Initialize mock and wrapper with the specific threshold
    mock_classifier = MockClassifier(input_logits, device=torch_device)
    wrapper = CORNClassifierWrapper(mock_classifier, threshold=threshold)

    # X is irrelevant for the mock
    X_dummy = None
    result_probs = wrapper.predict_proba(X_dummy)

    # The output is always NumPy on CPU for safe comparison
    assert np.array_equal(result_probs, expected_probs)
    assert result_probs.shape == expected_probs.shape
    assert result_probs.dtype == np.float32


# --- `predict` Tests ---
def test_predict_correct_argmax(torch_device):
    """Tests that predict returns the index with the highest probability
    using the default threshold (Class 2)."""

    # Logits for a 4-class sample (0 to 3).
    # Expected Class: 2 (using default threshold 0.5)
    input_logits = torch.tensor([[3.0, 3.0, -3.0]], dtype=torch.float32)

    mock_classifier = MockClassifier(input_logits, device=torch_device)
    wrapper = CORNClassifierWrapper(mock_classifier)  # Uses default threshold 0.5

    X_dummy = None
    predicted_labels = wrapper.predict(X_dummy)

    # The output of predict must be a NumPy array
    assert isinstance(predicted_labels, np.ndarray)
    assert predicted_labels.dtype == np.int64

    # The predicted class is 2
    assert np.array_equal(predicted_labels, np.array([2]))


def test_predict_with_custom_threshold(torch_device):
    """Tests that predict returns the correct class when using a
    custom threshold (Example 7 logic)."""

    # Logits from Case 7: CumProd P(y>=2)~0.9025
    input_logits = torch.tensor([[3.0, 3.0, -3.0]], dtype=torch.float32)

    # Threshold 0.92: P(y>=2) fails (0.9025 < 0.92). Prediction should drop to Class 1.
    custom_threshold = 0.92

    mock_classifier = MockClassifier(input_logits, device=torch_device)
    wrapper = CORNClassifierWrapper(mock_classifier, threshold=custom_threshold)

    X_dummy = None
    predicted_labels = wrapper.predict(X_dummy)

    # Expected class is 1
    assert np.array_equal(predicted_labels, np.array([1]))
