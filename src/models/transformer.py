import math
import os
import sys
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    TRANSFORMER_D_MODEL,
    TRANSFORMER_NHEAD,
    TRANSFORMER_LAYERS,
    TRANSFORMER_DIM_FF,
    TRANSFORMER_DROPOUT,
)


class _AttnCapturingLayer(nn.TransformerEncoderLayer):
    """TransformerEncoderLayer that stores attention weights on every forward pass."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_attn_weights: Optional[torch.Tensor] = None

    # Override the self-attention block to capture weights.
    # PyTorch 2.x signature: _sa_block(x, attn_mask, key_padding_mask, is_causal=False)
    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        attn_out, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        self._last_attn_weights = attn_weights.detach()
        return self.dropout1(attn_out)


class TransformerBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        d_model = TRANSFORMER_D_MODEL  # 64
        nhead = TRANSFORMER_NHEAD      # 8
        n_layers = TRANSFORMER_LAYERS  # 4
        dim_ff = TRANSFORMER_DIM_FF    # 256
        dropout = TRANSFORMER_DROPOUT  # 0.1

        self.input_proj = nn.Linear(1, d_model)

        self.encoder_layers = nn.ModuleList([
            _AttnCapturingLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layers[0],  # required arg; replaced below
            num_layers=1,                          # placeholder; we call layers manually
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Pre-compute sinusoidal PE for max sequence length 2001
        self.register_buffer("pe", self._build_pe(2001, d_model))

    @staticmethod
    def _build_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2001, 1)
        x = self.input_proj(x)           # (batch, 2001, 64)
        x = x + self.pe[:, : x.size(1)]  # add positional encoding

        for layer in self.encoder_layers:
            x = layer(x)                 # each layer stores its own attn weights

        x = x.mean(dim=1)               # global average pool → (batch, 64)
        x = self.fc(x)                  # (batch, 256)
        return x

    def get_attention_weights(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run a forward pass and return per-layer attention weight tensors."""
        with torch.no_grad():
            self.forward(x)
        return [
            layer._last_attn_weights
            for layer in self.encoder_layers
            if layer._last_attn_weights is not None
        ]
