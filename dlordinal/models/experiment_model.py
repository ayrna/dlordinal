from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ExperimentModel(nn.Module, metaclass=ABCMeta):
    """
    Base class for all experiment models.
    """

    features: nn.Module
    avgpool: nn.Module
    classifier: nn.Module

    @abstractmethod
    def scores(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the labels and probabilities for the given input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        """
        self.eval()
        x = self.scores(x)
        probas = F.softmax(x, dim=1)
        labels = probas.argmax(dim=1)
        labels, probas = map(lambda t: t.detach().cpu().numpy(), (labels, probas))
        return labels, probas

    def non_regularized_parameters(self) -> List[nn.parameter.Parameter]:
        """
        Get the non-regularized parameters.
        """
        return list(set(self.parameters()) - set(self.regularized_parameters()))

    @abstractmethod
    def regularized_parameters(self) -> List[nn.parameter.Parameter]:
        pass

    @abstractmethod
    def on_batch_end(self):
        pass
