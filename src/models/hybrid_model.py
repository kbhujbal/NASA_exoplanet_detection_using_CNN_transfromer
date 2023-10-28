import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.cnn_global import CNNGlobal
from src.models.cnn_local import CNNLocal
from src.models.transformer import TransformerBranch


class HybridExoplanetModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.cnn_global = CNNGlobal()        # outputs (batch, 512)
        self.cnn_local = CNNLocal()          # outputs (batch, 256)
        self.transformer = TransformerBranch()  # outputs (batch, 256)

        # 512 + 256 + 256 = 1024 combined features
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, batch_dict: dict) -> torch.Tensor:
        global_input = batch_dict["global"]  # (batch, 2001, 1)
        local_input = batch_dict["local"]    # (batch, 201, 1)

        feat_a = self.cnn_global(global_input)    # (batch, 512)
        feat_b = self.cnn_local(local_input)      # (batch, 256)
        feat_c = self.transformer(global_input)   # (batch, 256)

        combined = torch.cat([feat_a, feat_b, feat_c], dim=1)  # (batch, 1024)
        return self.classifier(combined)  # (batch, 1)

    def enable_dropout(self) -> None:
        """Set all Dropout layers to training mode for MC Dropout inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def get_transformer_attention(self) -> list:
        """Return attention weight tensors captured during the last forward pass."""
        return [
            layer._last_attn_weights
            for layer in self.transformer.encoder_layers
            if layer._last_attn_weights is not None
        ]
