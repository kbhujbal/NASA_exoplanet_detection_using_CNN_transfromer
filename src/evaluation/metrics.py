import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.hybrid_model import HybridExoplanetModel


def compute_all_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc_roc":           float(roc_auc_score(y_true, y_prob)),
        "f1":                float(f1_score(y_true, y_pred, zero_division=0)),
        "precision":         float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":            float(recall_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix":  confusion_matrix(y_true, y_pred),
    }


def ensemble_predict(
    model_paths: list,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    all_fold_probs = []

    for path in model_paths:
        model = HybridExoplanetModel().to(device)
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for batch in dataloader:
                global_x = batch["global"].to(device)
                local_x  = batch["local"].to(device)
                preds = model({"global": global_x, "local": local_x}).squeeze(1)
                fold_probs.extend(preds.cpu().numpy().tolist())

        all_fold_probs.append(fold_probs)

    # Average across folds → (N,)
    return np.array(all_fold_probs).mean(axis=0)
