import torch
from torch import Tensor


def h(x):
    return torch.log(1 + torch.exp(x))


class LogisticATLoss(torch.nn.Module):
    def __init__(self, num_classes, reg_lambda=0.0):
        self.num_classes = num_classes
        self.reg_lambda = reg_lambda
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        _, weights, x, thresholds = input
        projection = weights * x
        loss = 0
        for i in range(len(target)):
            for q in range(target[i].item()):
                loss += h(thresholds[q] - projection[i])
            for q in range(target[i].item(), self.num_classes - 1):
                loss += h(projection[i] - thresholds[q])

        loss = loss + ((self.reg_lambda / 2) * torch.pow(weights, 2))

        return loss
