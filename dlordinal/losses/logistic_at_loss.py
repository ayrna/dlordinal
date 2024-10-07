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

        if self.class_weights.size(0) != num_classes:
            raise ValueError(
                "class_weights must have the same number of elements as num_classes"
            )

        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        _, weights, x, thresholds = input
        projection = weights * x
        tiled_thresholds = torch.tile(thresholds, (target.shape[0], 1))
        tiled_projections = torch.tile(projection, (1, thresholds.shape[0]))
        sample_weights = self.class_weights[target]

        a = h(tiled_thresholds - tiled_projections)
        b = h(tiled_projections - tiled_thresholds)

        col_indices = torch.arange(thresholds.shape[0]).to(target.device)
        target_for_mask = target.unsqueeze(1)
        mask_for_a = col_indices < target_for_mask
        mask_for_b = col_indices >= target_for_mask

        a_masked = a * mask_for_a
        b_masked = b * mask_for_b

        a_weighted = a_masked * sample_weights.unsqueeze(1)
        b_weighted = b_masked * sample_weights.unsqueeze(1)

        loss = a_weighted.sum() + b_weighted.sum()

        # loss_l = 0
        # for i in range(len(target)):
        #     cw = self.class_weights[target[i]]
        #     for q in range(target[i].item()):
        #         loss_l += h(thresholds[q] - projection[i]) * cw
        #     for q in range(target[i].item(), self.num_classes - 1):
        #         loss_l += h(projection[i] - thresholds[q]) * cw

        # assert torch.allclose(loss, loss_l, rtol=1e-3)

        loss = loss + ((self.reg_lambda / 2) * torch.pow(weights, 2))

        return loss
