import torch
from torch.nn import Module
from torch.nn.functional import pad

# Esta es una forma de implementar la capa Lamda en PyTorch
# class Lambda(Module):
#     def __init__(self, func):
#         super().__init__()
#         self.func = func

#     def forward(self, x):
#         x = torch.clamp(x, 0.1, 0.9)

#         comp = 1.0 - x
#         # ciumprod is the cumulative product of the elements of the input tensor in the given dimension dim.
#         cumprod = torch.cumprod(comp, axis=1)
#         # Pads tensor
#         cumprod = pad(cumprod, [(0, 0), (1, 0)], mode='CONSTANT', constant_values=1.0)
#         x = pad(x, [(0, 0), (0, 1)], mode='CONSTANT', constant_values=1.0)
#         x = x * cumprod
        
#         return self.func(x)

class StickBreakingLayers(Module):
    def __init__(self, input_shape, num_classes) -> None:
        super().__init__()
        self.fcn1 = torch.nn.Linear(input_shape, num_classes)
        self.fcn2 = torch.nn.Sigmoid()

    def get_stick_probabilities(t):
        # Clamps all elements in input into the range [ min, max ]. Letting min_value and max_value be min and max, respectively 
        t = torch.clamp(t, 0.1, 0.9)

        comp = 1.0 - t
        # ciumprod is the cumulative product of the elements of the input tensor in the given dimension dim.
        cumprod = torch.cumprod(comp, axis=1)
        # Pads tensor
        cumprod = pad(cumprod, [(0, 0), (1, 0)], mode='CONSTANT', constant_values=1.0)
        t = pad(t, [(0, 0), (0, 1)], mode='CONSTANT', constant_values=1.0)
        return t * cumprod
    
    def forward(self,x) -> torch.Tensor:
        x = self.fcn1(x)
        x = self.fcn2(x)
        return self.get_stick_probabilities(x)