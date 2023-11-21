from typing import List

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch import nn
from torch.nn import functional as F

from ..experiment_model import ExperimentModel


class MockExperimentModel(ExperimentModel):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(10, 10)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(10, 2)

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def regularized_parameters(self) -> List[nn.parameter.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def on_batch_end(self):
        pass


def test_predict():
    model = MockExperimentModel()
    input_tensor = torch.randn((1, 10))
    labels, probas = model.predict(input_tensor)
    assert isinstance(labels, np.ndarray)
    assert isinstance(probas, np.ndarray)
    assert labels.shape[0] == probas.shape[0]


def test_non_regularized_parameters():
    model = MockExperimentModel()
    non_regularized_parameters = model.non_regularized_parameters()
    assert isinstance(non_regularized_parameters, list)
    assert all(
        isinstance(p, nn.parameter.Parameter) for p in non_regularized_parameters
    )


if __name__ == "__main__":
    test_predict()
    test_non_regularized_parameters()
