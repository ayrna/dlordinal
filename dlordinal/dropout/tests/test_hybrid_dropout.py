import pytest
import torch
from torch import nn

from dlordinal.dropout import HybridDropout, HybridDropoutContainer


class ModelWithHybridDropout(nn.Module):
    def __init__(self):
        super(ModelWithHybridDropout, self).__init__()
        self.fc1 = nn.Linear(10, 256)
        self.hybrid_dropout = HybridDropout()
        self.fc2 = nn.Linear(256, 5)  # Cambiar de 5 a la dimensión correcta

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.hybrid_dropout(x)
        x = self.fc2(x)
        return x


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def model():
    return ModelWithHybridDropout()


@pytest.fixture
def hybrid_dropout_container(model):
    return HybridDropoutContainer(model)


def test_set_targets(hybrid_dropout_container, device):
    targets = torch.tensor([1, 2, 3, 4, 5]).to(device)
    hybrid_dropout_container.set_targets(targets)

    hybrid_dropout_container = hybrid_dropout_container.to(device)

    hybrid_dropout = None
    for module in hybrid_dropout_container.model.modules():
        if isinstance(module, HybridDropout):
            hybrid_dropout = module
            hybrid_dropout = hybrid_dropout.to(device)
            break
    assert hybrid_dropout is not None
    assert torch.all(torch.eq(hybrid_dropout.batch_targets, targets))


def test_forward_with_targets(device):
    print(device)
    model = ModelWithHybridDropout().to(device)
    inputs = torch.randn(32, 10).to(device)
    targets = torch.randn(32).to(device)
    model.hybrid_dropout.batch_targets = targets
    outputs = model(inputs)

    # The number of classes is 5
    assert outputs.shape == (32, 5)


def test_forward_without_targets(hybrid_dropout_container, device):
    inputs = torch.randn(32, 10).to(device)

    with pytest.raises(ValueError):
        hybrid_dropout_container = hybrid_dropout_container.to(device)
        hybrid_dropout_container(inputs)
