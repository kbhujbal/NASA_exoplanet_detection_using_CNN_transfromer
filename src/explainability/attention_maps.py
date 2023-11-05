import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import OUTPUT_FIGURES_PATH


def extract_attention_weights(
    model: nn.Module,
    batch_dict: dict,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    global_x = batch_dict["global"].to(device)
    local_x = batch_dict["local"].to(device)

    with torch.no_grad():
        model({"global": global_x, "local": local_x})

    raw_weights = model.get_transformer_attention()
    # raw_weights: list of 4 tensors, each (batch, seq_len, seq_len)
    # if average_attn_weights=False was used, shape is (batch, nhead, seq_len, seq_len)

    processed = []
    for w in raw_weights:
        w = w.cpu().float()
        if w.dim() == 4:
            # (batch, nhead, seq_len, seq_len) → average over heads
            w = w.mean(dim=1)   # (batch, seq_len, seq_len)
        processed.append(w)

    # Stack layers and average → (batch, seq_len, seq_len)
    stacked = torch.stack(processed, dim=0).mean(dim=0)  # (batch, seq_len, seq_len)

    # Column sum: attention received by each position
    col_sum = stacked.sum(dim=1)  # (batch, seq_len)

    col_sum = col_sum.numpy()
    col_min = col_sum.min(axis=1, keepdims=True)
    col_max = col_sum.max(axis=1, keepdims=True)
    denom = np.where(col_max - col_min == 0, 1.0, col_max - col_min)
    normalized = (col_sum - col_min) / denom  # (batch, 2001)

    return normalized


def plot_attention_map(
    global_view: np.ndarray,
    attention_weights: np.ndarray,
    koi_id: str,
    prediction: str,
    save_path: str,
) -> None:
    # global_view: (2001, 1) or (2001,)
    flux = global_view.squeeze()
    phase = np.linspace(-0.5, 0.5, len(flux))
    attn = attention_weights.squeeze()   # (2001,)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(phase, flux, color="#1f77b4", linewidth=0.8)
    axes[0].set_ylabel("Normalized Flux")
    axes[0].set_title(f"KOI {koi_id} — Predicted: {prediction}", fontsize=13)

    im = axes[1].imshow(
        attn[np.newaxis, :],
        aspect="auto",
        extent=[-0.5, 0.5, 0, 1],
        cmap="hot",
        vmin=0,
        vmax=1,
    )
    axes[1].set_ylabel("Attention")
    axes[1].set_yticks([])
    axes[1].set_xlabel("Phase")
    fig.colorbar(im, ax=axes[1], orientation="vertical", fraction=0.02, pad=0.02)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
