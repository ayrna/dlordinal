from typing import Dict, Callable
from torch import nn

activation_function_by_name: Dict[str, Callable[[], nn.Module]] = {
    "relu": lambda: nn.ReLU(inplace=True),
    "elu": lambda: nn.ELU(inplace=True),
    "softplus": lambda: nn.Softplus(),
}
