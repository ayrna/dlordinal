from torch import nn
import torch

class OrdinalFullyConnected(nn.Module): # Esto habria que meterlo en la carpeta de layers y que no se le muestre al usuario pero si que se use internamente
    classifiers: nn.ModuleList
    
    def __init__(self, input_size: int, num_classes: int):
        super(OrdinalFullyConnected, self).__init__()
        self.classifiers = nn.ModuleList(
            [nn.Linear(input_size, 1) for _ in range(num_classes-1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = [classifier(x) for classifier in self.classifiers]
        x = torch.cat(xs, dim=1)
        x = torch.sigmoid(x)
        return x