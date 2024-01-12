import torch
from torch.nn import Module


class StickBreakingLayer(Module):
    """Base class to implement the stick breaking layer from :footcite:t:`liu2020unimodal`.

    Parameters
    ----------
    input_shape: int
        Input shape
    num_classes: int
        Number of classes
    """

    def __init__(self, input_shape: int, num_classes: int) -> None:
        super().__init__()
        self.fcn1 = torch.nn.Linear(input_shape, num_classes)
        self.fcn2 = torch.nn.Sigmoid()

    def get_stick_logits(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns:
        --------
        logits : torch.Tensor
            Logits of the stick breaking layer
        """

        # Clamps all elements in input into the range [ min, max ]. Letting min_value
        # and max_value be min and max, respectively
        x = torch.clamp(x, 0.1, 0.9)
        comp = 1.0 - x

        # cumprod is the cumulative product of the elements of the input tensor in
        # the given dimension dim.
        cumprod = torch.cumprod(comp, axis=1)
        logits = torch.log(x * cumprod)
        return logits

    def forward(self, x) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns:
        --------
        logits : torch.Tensor
            Logits of the stick breaking layer
        """

        x = self.fcn1(x)
        x = self.fcn2(x)
        logits = self.get_stick_logits(x)
        return logits
