import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import CNN_GLOBAL_FILTERS, CNN_GLOBAL_KERNEL, CNN_GLOBAL_POOL, DROPOUT_RATE


class CNNGlobal(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        f = CNN_GLOBAL_FILTERS  # [16, 32, 64, 128]
        k = CNN_GLOBAL_KERNEL   # 5
        p = k // 2              # 2  — same-length padding
        pool = CNN_GLOBAL_POOL  # 5

        self.conv_block = nn.Sequential(
            nn.Conv1d(1,     f[0], kernel_size=k, padding=p), nn.ReLU(), nn.MaxPool1d(pool),
            nn.Conv1d(f[0], f[1], kernel_size=k, padding=p), nn.ReLU(), nn.MaxPool1d(pool),
            nn.Conv1d(f[1], f[2], kernel_size=k, padding=p), nn.ReLU(), nn.MaxPool1d(pool),
            nn.Conv1d(f[2], f[3], kernel_size=k, padding=p), nn.ReLU(), nn.MaxPool1d(pool),
        )

        flat_size = self._get_flat_size()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )

    def _get_flat_size(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 2001)
            out = self.conv_block(dummy)
            return out.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2001, 1) → (batch, 1, 2001)
        x = x.transpose(1, 2)
        x = self.conv_block(x)
        x = self.fc(x)
        return x  # (batch, 512)
