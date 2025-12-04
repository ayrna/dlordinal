from collections import Counter
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SORDLoss(nn.Module):
    def __init__(
        self,
        alpha: float,
        num_classes: int,
        train_targets: Tensor,
        prox: bool = False,
        ftype: str = "max",
        weight: Optional[torch.Tensor] = None,
        use_logits: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.prox = prox
        self.ftype = ftype
        self.use_logits = use_logits
        self.train_targets = train_targets
        self.weight = weight

        # Initialize proximity matrix if needed
        if self.prox:
            self.class_counts_dict = self._create_classcounts_dict(train_targets)
            self.prox_mat = create_prox_mat(self.class_counts_dict, inv=False)
            if not hasattr(self, "prox_mat"):
                self.register_buffer("prox_mat", self.prox_mat)
            self.norm_prox_mat = F.normalize(self.prox_mat, p=1, dim=0)
            if not hasattr(self, "norm_prox_mat"):
                self.register_buffer("norm_prox_mat", self.norm_prox_mat)

    def _create_classcounts_dict(self, targets):
        class_counts = Counter(np.array(targets))
        class_counts_dict = {i: class_counts.get(i, 0) for i in range(self.num_classes)}
        return class_counts_dict

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.use_logits:
            input_logprob = F.log_softmax(input, dim=1)
        else:
            input_logprob = torch.log(input + 1e-9)

        if self.prox:
            self.prox_mat = self.prox_mat.to(input.device)
            self.norm_prox_mat = self.norm_prox_mat.to(input.device)

        if not self.prox:
            phi = torch.abs(
                torch.arange(self.num_classes, device=input.device).view(1, -1)
                - target.double().view(-1, 1)
            )
        else:
            if self.ftype == "max":
                phi = torch.max(self.prox_mat) - self.prox_mat[target]
            elif self.ftype == "norm_max":
                phi = torch.max(self.norm_prox_mat) - self.norm_prox_mat[target]
            elif self.ftype == "norm_log":
                phi = -torch.log(self.norm_prox_mat[target])
            elif self.ftype == "log":
                phi = -torch.log(self.prox_mat[target])
            elif self.ftype == "norm_division":
                phi = 1.0 / (self.norm_prox_mat[target])
            elif self.ftype == "division":
                phi = 1.0 / (self.prox_mat[target])

        softmax_targets = F.softmax(-self.alpha * phi, dim=1)

        per_sample_loss = -torch.sum(softmax_targets * input_logprob, dim=1)  # [batch]

        # Class weight
        if self.weight is not None:
            # self.weight: [num_classes]
            # target: [batch]
            sample_weights = self.weight[target].to(input.device)  # [batch]
            per_sample_loss = per_sample_loss * sample_weights

        return per_sample_loss.mean()


def create_prox_mat(dist_dict, inv=True):
    labels = list(dist_dict.keys())
    labels.sort()
    denominator = sum(dist_dict.values())
    prox_mat = np.zeros([len(labels), len(labels)])
    for label1 in labels:
        for label2 in labels:
            label1 = int(label1)
            label2 = int(label2)
            minlabel, maxlabel = min(label1, label2), max(label1, label2)
            numerator = dist_dict[label1] / 2
            if minlabel == label1:  # Above the diagonal
                for tmp_label in range(minlabel + 1, maxlabel + 1):
                    numerator += dist_dict[tmp_label]
            else:  # Under the diagonal
                for tmp_label in range(maxlabel - 1, minlabel - 1, -1):
                    numerator += dist_dict[tmp_label]
            if inv:
                prox_mat[label1][label2] = (-np.log(numerator / denominator)) ** -1
            else:
                prox_mat[label1][label2] = -np.log(numerator / denominator)
    return torch.tensor(prox_mat)
