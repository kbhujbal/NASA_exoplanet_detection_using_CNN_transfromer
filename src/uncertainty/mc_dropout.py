import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import MC_DROPOUT_PASSES, UNCERTAINTY_HIGH, UNCERTAINTY_MEDIUM


def mc_dropout_predict(
    model: nn.Module,
    batch_dict: dict,
    n_passes: int = MC_DROPOUT_PASSES,
    device: torch.device = torch.device("cpu"),
) -> dict:
    model.eval()
    model.enable_dropout()

    global_x = batch_dict["global"].to(device)
    local_x = batch_dict["local"].to(device)

    pass_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            out = model({"global": global_x, "local": local_x})
            pass_probs.append(out.squeeze(1).cpu().numpy())

    # pass_probs: list of n_passes arrays, each shape (batch,)
    pass_probs = np.stack(pass_probs, axis=0)  # (n_passes, batch)

    mean_prob = float(pass_probs.mean(axis=0).item() if pass_probs.shape[1] == 1
                      else pass_probs.mean(axis=0).squeeze())
    uncertainty = float(pass_probs.std(axis=0).item() if pass_probs.shape[1] == 1
                        else pass_probs.std(axis=0).squeeze())

    prediction = "CONFIRMED" if mean_prob >= 0.5 else "FALSE POSITIVE"
    confidence_pct = mean_prob * 100 if prediction == "CONFIRMED" else (1 - mean_prob) * 100

    if uncertainty < UNCERTAINTY_HIGH:
        confidence_level = "HIGH"
    elif uncertainty < UNCERTAINTY_MEDIUM:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    return {
        "prediction": prediction,
        "mean_probability": mean_prob,
        "uncertainty": uncertainty,
        "confidence_pct": confidence_pct,
        "confidence_level": confidence_level,
        "all_passes": pass_probs.flatten().tolist(),
    }


def batch_mc_predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    records = []

    for batch_idx, batch in enumerate(dataloader):
        global_x = batch["global"]
        local_x = batch["local"]
        true_labels = batch["label"].numpy()

        batch_size = global_x.shape[0]
        for sample_i in range(batch_size):
            single_dict = {
                "global": global_x[sample_i].unsqueeze(0),
                "local": local_x[sample_i].unsqueeze(0),
            }
            result = mc_dropout_predict(model, single_dict, n_passes=MC_DROPOUT_PASSES, device=device)
            records.append({
                "index": batch_idx * dataloader.batch_size + sample_i,
                "prediction": result["prediction"],
                "mean_probability": result["mean_probability"],
                "uncertainty": result["uncertainty"],
                "confidence_pct": result["confidence_pct"],
                "confidence_level": result["confidence_level"],
                "true_label": int(true_labels[sample_i]),
            })

    return pd.DataFrame(records)


def print_prediction_card(koi_id: str, result: dict) -> None:
    pred = result["prediction"]
    conf_pct = result["confidence_pct"]
    unc_pct = result["uncertainty"] * 100
    level = result["confidence_level"]

    width = 47
    border = "─" * width

    lines = [
        f"┌{border}┐",
        f"│  KEPLER OBJECT: {koi_id:<28}│",
        f"│  PREDICTION:    {pred:<28}│",
        f"│  CONFIDENCE:    {conf_pct:.1f}%  ±  {unc_pct:.1f}%{'':<17}│",
        f"│  CERTAINTY:     {level:<28}│",
        f"└{border}┘",
    ]
    print("\n".join(lines))
