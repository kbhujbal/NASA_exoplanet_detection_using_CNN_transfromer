import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    OUTPUT_MODELS_PATH,
    N_FOLDS,
    BATCH_SIZE,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    RANDOM_SEED,
)
from src.training.losses import FocalLoss
from src.training.dataset import get_dataloaders
from src.models.hybrid_model import HybridExoplanetModel

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        fold_idx: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.fold_idx = fold_idx
        self.device = device

    def train(self, max_epochs: int, patience: int) -> dict:
        optimizer = AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
        criterion = FocalLoss()

        os.makedirs(OUTPUT_MODELS_PATH, exist_ok=True)
        checkpoint_path = os.path.join(OUTPUT_MODELS_PATH, f"fold_{self.fold_idx}_best.pt")

        history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": []}
        best_val_auc = -1.0
        epochs_without_improvement = 0

        for epoch in range(1, max_epochs + 1):
            # ── Training ────────────────────────────────────────────────────
            self.model.train()
            train_losses = []

            for batch in self.train_loader:
                global_x = batch["global"].to(self.device)
                local_x = batch["local"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                preds = self.model({"global": global_x, "local": local_x})
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # ── Validation ──────────────────────────────────────────────────
            self.model.eval()
            val_losses = []
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in self.val_loader:
                    global_x = batch["global"].to(self.device)
                    local_x = batch["local"].to(self.device)
                    labels = batch["label"].to(self.device)

                    preds = self.model({"global": global_x, "local": local_x})
                    loss = criterion(preds, labels)
                    val_losses.append(loss.item())

                    all_preds.extend(preds.squeeze(1).cpu().numpy().tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())

            scheduler.step()

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))

            all_preds_arr = np.array(all_preds)
            all_labels_arr = np.array(all_labels, dtype=int)

            val_auc = roc_auc_score(all_labels_arr, all_preds_arr)
            val_f1 = f1_score(all_labels_arr, (all_preds_arr >= 0.5).astype(int), zero_division=0)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)
            history["val_f1"].append(val_f1)

            logger.info(
                "Fold %d | Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | "
                "val_auc=%.4f | val_f1=%.4f",
                self.fold_idx, epoch, max_epochs, train_loss, val_loss, val_auc, val_f1,
            )

            # ── Checkpoint & early stopping ─────────────────────────────────
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_auc": val_auc,
                        "val_f1": val_f1,
                    },
                    checkpoint_path,
                )
                logger.info("  → New best checkpoint saved (val_auc=%.4f)", best_val_auc)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs).",
                        epoch, patience,
                    )
                    break

        return history


def run_kfold_training(device: torch.device) -> list[dict]:
    all_histories = []

    for fold_idx in range(N_FOLDS):
        logger.info("=" * 60)
        logger.info("Starting fold %d / %d", fold_idx + 1, N_FOLDS)
        logger.info("=" * 60)

        train_loader, val_loader = get_dataloaders(fold_idx, N_FOLDS, BATCH_SIZE)

        model = HybridExoplanetModel().to(device)

        trainer = Trainer(model, train_loader, val_loader, fold_idx, device)
        history = trainer.train(MAX_EPOCHS, EARLY_STOPPING_PATIENCE)
        all_histories.append(history)

    best_aucs = [max(h["val_auc"]) for h in all_histories]
    mean_auc = float(np.mean(best_aucs))
    std_auc = float(np.std(best_aucs))

    print(f"\nK-Fold Training Summary")
    print(f"  Best val AUC per fold : {[f'{a:.4f}' for a in best_aucs]}")
    print(f"  Mean ± Std val AUC    : {mean_auc:.4f} ± {std_auc:.4f}")

    return all_histories
