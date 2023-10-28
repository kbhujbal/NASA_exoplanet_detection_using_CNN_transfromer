import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import CNN_LOCAL_FILTERS, CNN_LOCAL_KERNEL, CNN_LOCAL_POOL, DROPOUT_RATE


class CNNLocal(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        f = CNN_LOCAL_FILTERS  # [16, 32, 64]
        k = CNN_LOCAL_KERNEL   # 3
        p = k // 2             # 1  — same-length padding
        pool = CNN_LOCAL_POOL  # 3

        self.conv_block = nn.Sequential(
            nn.Conv1d(1,     f[0], kernel_size=k, padding=p), nn.ReLU(), nn.MaxPool1d(pool),
            nn.Conv1d(f[0], f[1], kernel_size=k, padding=p), nn.ReLU(), nn.MaxPool1d(pool),
            nn.Conv1d(f[1], f[2], kernel_size=k, padding=p), nn.ReLU(), nn.MaxPool1d(pool),
        )

        flat_size = self._get_flat_size()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )

    def _get_flat_size(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 201)
            out = self.conv_block(dummy)
            return out.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 201, 1) → (batch, 1, 201)
        x = x.transpose(1, 2)
        x = self.conv_block(x)
        x = self.fc(x)
        return x  # (batch, 256)
