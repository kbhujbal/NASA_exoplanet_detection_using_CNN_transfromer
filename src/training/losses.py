import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import FOCAL_ALPHA, FOCAL_GAMMA


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # predictions: (batch, 1) sigmoid outputs → squeeze to (batch,)
        p = predictions.squeeze(1).clamp(min=1e-7, max=1 - 1e-7)

        p_t = torch.where(targets == 1.0, p, 1.0 - p)
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = -self.alpha * focal_weight * torch.log(p_t)
        return loss.mean()
