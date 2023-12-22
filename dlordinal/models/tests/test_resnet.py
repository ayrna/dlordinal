import pytest
import torch
import torchvision.models as models

from ..obd_ecoc import (
    OBDECOCModel,
)


@pytest.fixture
def sample_tensor():
    return torch.randn(1, 3, 224, 224)


def test_OBDECOCModel(sample_tensor):
    resnet_ecoc = OBDECOCModel(
        num_classes=6,
        base_classifier=models.resnet18(num_classes=100),
        base_n_outputs=100,
    )

    assert isinstance(resnet_ecoc, OBDECOCModel)
    output_tensor = resnet_ecoc(sample_tensor)
    assert output_tensor.shape[1] == resnet_ecoc.num_classes - 1
