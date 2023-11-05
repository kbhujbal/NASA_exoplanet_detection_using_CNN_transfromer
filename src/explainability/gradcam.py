import os
import sys
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import OUTPUT_FIGURES_PATH


def compute_gradcam(
    model: nn.Module,
    batch_dict: dict,
    target_class: int = 1,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    model.eval()

    # Locate the last Conv1d in cnn_global.conv_block
    # Sequential layout: (Conv,ReLU,Pool) × 4 — last Conv1d is at index 9
    last_conv: nn.Conv1d = model.cnn_global.conv_block[9]

    feature_maps: Optional[torch.Tensor] = None
    gradients: Optional[torch.Tensor] = None

    def fwd_hook(module, input, output):
        nonlocal feature_maps
        feature_maps = output  # (batch, channels, length)

    def bwd_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]  # (batch, channels, length)

    fwd_handle = last_conv.register_forward_hook(fwd_hook)
    bwd_handle = last_conv.register_full_backward_hook(bwd_hook)

    global_x = batch_dict["global"].to(device).requires_grad_(False)
    local_x = batch_dict["local"].to(device).requires_grad_(False)

    # Need gradients w.r.t. feature maps, not inputs
    model.zero_grad()
    preds = model({"global": global_x, "local": local_x})  # (batch, 1)

    # Build target score: use raw prediction for target_class
    if target_class == 1:
        score = preds.squeeze(1)        # (batch,)
    else:
        score = 1.0 - preds.squeeze(1)

    score.sum().backward()

    fwd_handle.remove()
    bwd_handle.remove()

    # Grad-CAM computation
    # gradients, feature_maps: (batch, channels, length)
    weights = gradients.mean(dim=2)         # global avg pool → (batch, channels)
    batch_size, channels, length = feature_maps.shape

    cam = torch.zeros(batch_size, length, device=device)
    for c in range(channels):
        cam += weights[:, c].unsqueeze(1) * feature_maps[:, c, :]  # (batch, length)

    cam = F.relu(cam)                        # (batch, length)

    # Upsample to 2001 via linear interpolation
    cam = cam.unsqueeze(1)                   # (batch, 1, length)
    cam = F.interpolate(cam, size=2001, mode="linear", align_corners=False)
    cam = cam.squeeze(1)                     # (batch, 2001)

    cam = cam.detach().cpu().numpy()

    # Normalize each sample to [0, 1]
    cam_min = cam.min(axis=1, keepdims=True)
    cam_max = cam.max(axis=1, keepdims=True)
    denom = np.where(cam_max - cam_min == 0, 1.0, cam_max - cam_min)
    cam = (cam - cam_min) / denom

    return cam


def plot_gradcam(
    global_view: np.ndarray,
    gradcam: np.ndarray,
    koi_id: str,
    prediction: str,
    save_path: str,
) -> None:
    flux = global_view.squeeze()
    phase = np.linspace(-0.5, 0.5, len(flux))
    cam = gradcam.squeeze()  # (2001,)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(phase, flux, color="#1f77b4", linewidth=0.8)
    axes[0].set_ylabel("Normalized Flux")
    axes[0].set_title(f"KOI {koi_id} — Grad-CAM — Predicted: {prediction}", fontsize=13)

    im = axes[1].imshow(
        cam[np.newaxis, :],
        aspect="auto",
        extent=[-0.5, 0.5, 0, 1],
        cmap="jet",
        vmin=0,
        vmax=1,
    )
    axes[1].set_ylabel("Grad-CAM")
    axes[1].set_yticks([])
    axes[1].set_xlabel("Phase")
    fig.colorbar(im, ax=axes[1], orientation="vertical", fraction=0.02, pad=0.02)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
