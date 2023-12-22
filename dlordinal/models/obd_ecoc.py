from torch import Tensor
import numpy as np
import torch

import torch.nn as nn
from collections import OrderedDict, namedtuple


class OBDECOCModel(nn.Module):
    def __init__(
        self, num_classes: int, base_classifier: nn.Module, base_n_outputs: int
    ) -> None:
        super().__init__()
        self.base_classifier = base_classifier
        self.obd_output = nn.Sequential(
            OrderedDict(
                [
                    ("penultimate_activation", nn.ReLU()),
                    ("last_linear", nn.Linear(base_n_outputs, num_classes - 1)),
                    ("last_activation", nn.Sigmoid()),
                ]
            )
        )
        self.transformer = ECOCOutputTransformer(num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.base_classifier(x)
        x = self.obd_output(x)
        return x

    def predict_from_inputs(self, x):
        raw_output = self(x)
        return PredictOutput(
            self.transformer.scores(raw_output),
            self.transformer.probas(raw_output),
            self.transformer.labels(raw_output),
        )


PredictOutput = namedtuple("PredictOutput", ["scores", "probas", "label"])


class ECOCOutputTransformer(nn.Module):
    target_class: Tensor

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.num_classes = n_classes
        target_class = np.ones((n_classes, n_classes - 1), dtype=np.float32)
        target_class[np.triu_indices(n_classes, 0, n_classes - 1)] = 0.0
        target_class = torch.tensor(target_class, dtype=torch.float32)
        self.register_buffer("target_class", target_class)

    def probas(self, output):
        return torch.softmax(self.scores(output), dim=1)

    def scores(self, output):
        return -torch.cdist(output, self.target_class)

    def labels(self, output):
        scores = self.scores(output)
        return scores.argmax(dim=1)
