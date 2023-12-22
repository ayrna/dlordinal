from torch import Tensor
import numpy as np
import torch

import torch.nn as nn
from collections import OrderedDict, namedtuple


class OBDECOCModel(nn.Module):
    """Ordinal Binary Decomposition (OBD) wrapper model from :footcite:t:`barbero2023error`.
    It transforms the output of the provided base classifier into
    ``num_classes - 1`` outputs bounded between 0 and 1, representing
    the class threshold probabilities :math:`P(y > q)`.

    Parameters
    ----------
    num_classes : int
        Number of classes.
    base_classifier: nn.Module
        Base classifier that will be wrapped.
    base_n_outputs:
        Number of outputs of the base classifier. The models implemented
        in ``torchvision`` have 1000 as the default
    """

    num_classes: int

    def __init__(
        self, num_classes: int, base_classifier: nn.Module, base_n_outputs: int
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
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
        """
        Parameters
        ----------
        x : torch.Tensor
            Input to the model

        Returns
        -------
        threshold_probas : torch.Tensor
            Predicted threshold probabilities
        """
        x = self.base_classifier(x)
        x = self.obd_output(x)
        return x

    def predict_from_inputs(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input to the model

        Returns
        -------
        An object with the following attributes.

        scores : Tensor
            The negative distance to each class ideal vector, to use
            as class scores.
        probas : Tensor
            The predicted probability of belonging to each class :math:`P(y = q)`.
        labels : Tensor
            The predicted integer label according to the ECOC assignment
            scheme.
        """
        raw_output = self(x)
        return PredictOutput(
            self.transformer.scores(raw_output),
            self.transformer.probas(raw_output),
            self.transformer.labels(raw_output),
        )


PredictOutput = namedtuple("PredictOutput", ["scores", "probas", "label"])


class ECOCOutputTransformer(nn.Module):
    """A transformer for the output of the OBD model in order
    to apply the ECOC scheme.

    Parameters
    ----------
    num_classes : int
        Number of classes.
    """

    target_class: Tensor

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        target_class = np.ones((num_classes, num_classes - 1), dtype=np.float32)
        target_class[np.triu_indices(num_classes, 0, num_classes - 1)] = 0.0
        target_class = torch.tensor(target_class, dtype=torch.float32)
        self.register_buffer("target_class", target_class)

    def probas(self, output):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input to the model

        Returns
        -------
        probas : Tensor
            The predicted probability of belonging to each class :math:`P(y = q)`.
        """
        return torch.softmax(self.scores(output), dim=1)

    def scores(self, output):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input to the model

        Returns
        -------
        scores : Tensor
            The negative distance to each class ideal vector, to use
            as class scores.
        """
        return -torch.cdist(output, self.target_class)

    def labels(self, output):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input to the model

        Returns
        -------
        labels : Tensor
            The predicted integer label according to the ECOC assignment
            scheme.
        """
        scores = self.scores(output)
        return scores.argmax(dim=1)
