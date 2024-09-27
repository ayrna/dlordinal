import torch
from torch import Tensor


def h(x):
    return torch.log(1 + torch.exp(x))


class LogisticATLoss(torch.nn.Module):
    def __init__(self, num_classes, reg_lambda=0.0, class_weights=None):
        self.num_classes = num_classes
        self.reg_lambda = reg_lambda
        self.class_weights = class_weights

        if class_weights is None:
            self.class_weights = torch.ones(num_classes)

        if isinstance(self.class_weights, list):
            self.class_weights = torch.tensor(self.class_weights)

        if not isinstance(self.class_weights, Tensor):
            raise ValueError("class_weights must be a list or a Tensor")

        super().__init__()

    # def forward(self, input: Tensor, target: Tensor) -> Tensor:
    #     _, weights, x, thresholds = input
    #     projection = weights * x
    #     loss = 0
    #     for i in range(len(target)):
    #         cw = self.class_weights[target[i]]
    #         for q in range(target[i].item()):
    #             loss += h(thresholds[q] - projection[i]) * cw
    #         for q in range(target[i].item(), self.num_classes - 1):
    #             loss += h(projection[i] - thresholds[q]) * cw

    #     loss = loss + ((self.reg_lambda / 2) * torch.pow(weights, 2))

    #     return loss

    # def forward(self, input: Tensor, target: Tensor) -> Tensor:
    #     _, weights, x, thresholds = input
    #     projection = weights * x

    #     # Create masks for the loss calculation
    #     target_expanded = target.unsqueeze(1).expand(-1, self.num_classes - 1)
    #     thresholds_expanded = thresholds.unsqueeze(0).expand(target.size(0), -1)
    #     projection_expanded = projection.unsqueeze(1).expand(-1, self.num_classes - 1)

    #     # Calculate the loss for q < target
    #     mask_lt = (
    #         torch.arange(self.num_classes - 1).unsqueeze(0).expand(target.size(0), -1)
    #         < target_expanded
    #     )
    #     loss_lt = h(thresholds_expanded - projection_expanded) * mask_lt.float()

    #     # Calculate the loss for q >= target
    #     mask_ge = (
    #         torch.arange(self.num_classes - 1).unsqueeze(0).expand(target.size(0), -1)
    #         >= target_expanded
    #     )
    #     loss_ge = h(projection_expanded - thresholds_expanded) * mask_ge.float()

    #     # Combine the losses and apply class weights
    #     loss = (loss_lt + loss_ge) * self.class_weights[target].unsqueeze(1)

    #     # Sum the losses and add regularization term
    #     loss = loss.sum() + ((self.reg_lambda / 2) * torch.pow(weights, 2))

    #     return loss

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        _, weights, x, thresholds = input
        projection = weights * x

        # Precompute class masks
        class_range = torch.arange(
            self.num_classes - 1, device=target.device
        ).unsqueeze(0)
        target_expanded = target.unsqueeze(1)

        mask_lt = class_range < target_expanded  # q < target
        mask_ge = class_range >= target_expanded  # q >= target

        # Compute loss terms
        thresholds_expanded = thresholds.unsqueeze(0)
        projection_expanded = projection.unsqueeze(1)

        # Use broadcasting for the loss terms without explicit expansion
        loss_lt = h(thresholds_expanded - projection_expanded) * mask_lt.float()
        loss_ge = h(projection_expanded - thresholds_expanded) * mask_ge.float()

        # Combine losses and apply class weights directly
        class_weighted_loss = (loss_lt + loss_ge) * self.class_weights[
            target
        ].unsqueeze(1)

        # Final loss summation with regularization term
        total_loss = (
            class_weighted_loss.sum()
            + (self.reg_lambda / 2) * torch.pow(weights, 2).sum()
        )

        return total_loss
