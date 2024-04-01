import pytest
import torch

from dlordinal.dropout import HybridDropout


@pytest.fixture
def batch_targets():
    return torch.randn(10)  # Assuming batch size is 10


def test_constructor_valid_arguments(batch_targets):
    dropout = HybridDropout(batch_targets=batch_targets, p=0.5, beta=0.1)
    assert dropout.p == 0.5
    assert dropout.beta == 0.1
    assert torch.all(torch.eq(dropout.batch_targets, batch_targets))


def test_constructor_invalid_p_value(batch_targets):
    with pytest.raises(ValueError):
        HybridDropout(batch_targets=batch_targets, p=1.5)


def test_forward_shape(batch_targets):
    dropout = HybridDropout(batch_targets=batch_targets)
    input_tensor = torch.randn(5, 10)  # Assuming input tensor shape is (5, 10)
    output = dropout(input_tensor)
    assert output.shape == input_tensor.shape


# def test_forward_non_training(batch_targets):
#     dropout = HybridDropout(batch_targets=batch_targets)
#     input_tensor = torch.randn(5, 10)  # Assuming input tensor shape is (5, 10)
#     output = dropout(input_tensor)
#     assert torch.all(torch.eq(output, input_tensor))


# You can add more tests as needed
