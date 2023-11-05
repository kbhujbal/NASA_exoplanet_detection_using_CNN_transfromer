import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import shap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import OUTPUT_FIGURES_PATH

_GLOBAL_LEN = 2001
_LOCAL_LEN = 201
_FLAT_LEN = _GLOBAL_LEN + _LOCAL_LEN  # 2202


class _FlatWrapper(nn.Module):
    """Accepts flat (batch, 2202) input, routes through the full hybrid model."""

    def __init__(self, hybrid_model: nn.Module) -> None:
        super().__init__()
        self.model = hybrid_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2202)
        global_flat = x[:, :_GLOBAL_LEN]           # (batch, 2001)
        local_flat = x[:, _GLOBAL_LEN:]             # (batch, 201)
        global_view = global_flat.unsqueeze(2)      # (batch, 2001, 1)
        local_view = local_flat.unsqueeze(2)        # (batch, 201, 1)
        return self.model({"global": global_view, "local": local_view})


def _collect_flat_samples(dataloader: DataLoader, n: int, device: torch.device) -> torch.Tensor:
    """Pull up to n samples from a dataloader, flatten each, return tensor (n, 2202)."""
    collected = []
    for batch in dataloader:
        global_x = batch["global"]  # (bs, 2001, 1)
        local_x = batch["local"]    # (bs, 201, 1)
        g_flat = global_x.squeeze(2)   # (bs, 2001)
        l_flat = local_x.squeeze(2)    # (bs, 201)
        flat = torch.cat([g_flat, l_flat], dim=1)  # (bs, 2202)
        collected.append(flat)
        if sum(t.shape[0] for t in collected) >= n:
            break
    combined = torch.cat(collected, dim=0)[:n]
    return combined.to(device).float()


def compute_shap_values(
    model: nn.Module,
    background_loader: DataLoader,
    test_loader: DataLoader,
    n_background: int = 100,
    n_test: int = 200,
    device: torch.device = torch.device("cpu"),
) -> dict:
    model.eval()

    wrapper = _FlatWrapper(model).to(device)
    wrapper.eval()

    background_data = _collect_flat_samples(background_loader, n_background, device)
    test_data = _collect_flat_samples(test_loader, n_test, device)

    explainer = shap.DeepExplainer(wrapper, background_data)
    shap_values = explainer.shap_values(test_data)

    # shap_values may be a list (one array per output) or a single array
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    feature_names = (
        [f"global_t{i}" for i in range(_GLOBAL_LEN)]
        + [f"local_t{i}" for i in range(_LOCAL_LEN)]
    )

    return {
        "shap_values": np.array(shap_values),
        "test_inputs": test_data.cpu().numpy(),
        "feature_names": feature_names,
    }


def plot_shap_summary(shap_dict: dict, save_path: str) -> None:
    shap_values = shap_dict["shap_values"]    # (n_test, 2202)
    test_inputs = shap_dict["test_inputs"]    # (n_test, 2202)
    feature_names = shap_dict["feature_names"]

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        test_inputs,
        feature_names=feature_names,
        max_display=30,
        plot_type="dot",
        show=False,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close("all")


def plot_shap_waterfall(
    shap_dict: dict,
    sample_idx: int,
    koi_id: str,
    prediction: str,
    save_path: str,
) -> None:
    shap_values = shap_dict["shap_values"]    # (n_test, 2202)
    test_inputs = shap_dict["test_inputs"]    # (n_test, 2202)
    feature_names = shap_dict["feature_names"]

    expected_value = float(shap_values.mean())

    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=expected_value,
        data=test_inputs[sample_idx],
        feature_names=feature_names,
    )

    shap.waterfall_plot(explanation, max_display=20, show=False)

    fig = plt.gcf()
    fig.suptitle(f"KOI {koi_id} — SHAP Explanation — {prediction}", fontsize=12, y=1.01)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close("all")
