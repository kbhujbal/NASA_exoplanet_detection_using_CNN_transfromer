import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    DATA_PROCESSED_PATH,
    OUTPUT_RESULTS_PATH,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BATCH_SIZE,
    RANDOM_SEED,
    N_FOLDS,
)
from src.training.losses import FocalLoss
from src.training.dataset import KeplerDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_MAX_EPOCHS_BASELINE = 50
_THRESHOLD = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Shared data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_fold0():
    global_views = np.load(os.path.join(DATA_PROCESSED_PATH, "global_views.npy"))
    local_views = np.load(os.path.join(DATA_PROCESSED_PATH, "local_views.npy"))
    labels = np.load(os.path.join(DATA_PROCESSED_PATH, "labels.npy"))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    train_idx, val_idx = list(skf.split(global_views, labels))[0]

    return (
        global_views[train_idx], local_views[train_idx], labels[train_idx],
        global_views[val_idx],   local_views[val_idx],   labels[val_idx],
    )


def _metrics(y_true, probs) -> dict:
    preds = (probs >= _THRESHOLD).astype(int)
    return {
        "AUC-ROC":   round(roc_auc_score(y_true, probs), 4),
        "F1":        round(f1_score(y_true, preds, zero_division=0), 4),
        "Precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, preds, zero_division=0), 4),
        "AP":        round(average_precision_score(y_true, probs), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shared PyTorch training loop
# ─────────────────────────────────────────────────────────────────────────────

def _train_pytorch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    input_key: str = "global",
) -> np.ndarray:
    """Train model for up to _MAX_EPOCHS_BASELINE epochs; return val probabilities."""
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss()

    for epoch in range(1, _MAX_EPOCHS_BASELINE + 1):
        model.train()
        for batch in train_loader:
            x = batch[input_key].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            logger.info("  epoch %d/%d", epoch, _MAX_EPOCHS_BASELINE)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch[input_key].to(device)
            probs = model(x).squeeze(1).cpu().numpy()
            all_probs.extend(probs.tolist())
    return np.array(all_probs)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1 – Random Forest
# ─────────────────────────────────────────────────────────────────────────────

def _run_random_forest(g_tr, l_tr, y_tr, g_val, l_val, y_val) -> dict:
    logger.info("Training Random Forest …")
    X_train = np.concatenate(
        [g_tr.reshape(len(g_tr), -1), l_tr.reshape(len(l_tr), -1)], axis=1
    )
    X_val = np.concatenate(
        [g_val.reshape(len(g_val), -1), l_val.reshape(len(l_val), -1)], axis=1
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train, y_tr)
    probs = clf.predict_proba(X_val)[:, 1]
    return _metrics(y_val, probs)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2 – Vanilla 1D CNN (global view only)
# ─────────────────────────────────────────────────────────────────────────────

class _VanillaCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 16,  kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(5),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(5),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(5),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(5),
        )
        flat = self._flat_size()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1),    nn.Sigmoid(),
        )

    def _flat_size(self) -> int:
        with torch.no_grad():
            return self.conv_block(torch.zeros(1, 1, 2001)).numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2001, 1) → (batch, 1, 2001)
        return self.head(self.conv_block(x.transpose(1, 2)))


def _run_vanilla_cnn(g_tr, l_tr, y_tr, g_val, l_val, y_val, device) -> dict:
    logger.info("Training Vanilla CNN …")
    train_ds = KeplerDataset(g_tr, l_tr, y_tr, augment=False)
    val_ds   = KeplerDataset(g_val, l_val, y_val, augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = _VanillaCNN()
    probs = _train_pytorch(model, train_dl, val_dl, device, input_key="global")
    return _metrics(y_val, probs)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 3 – LSTM
# ─────────────────────────────────────────────────────────────────────────────

class _LSTMModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        # bidirectional → 2 * 128 = 256
        self.head = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2001, 1)
        _, (hidden, _) = self.lstm(x)
        # hidden: (num_layers * 2, batch, 128)
        # take last layer's forward + backward hidden states
        fwd = hidden[-2]   # (batch, 128)
        bwd = hidden[-1]   # (batch, 128)
        last_hidden = torch.cat([fwd, bwd], dim=1)  # (batch, 256)
        return self.head(last_hidden)


def _run_lstm(g_tr, l_tr, y_tr, g_val, l_val, y_val, device) -> dict:
    logger.info("Training LSTM …")
    train_ds = KeplerDataset(g_tr, l_tr, y_tr, augment=False)
    val_ds   = KeplerDataset(g_val, l_val, y_val, augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = _LSTMModel()
    probs = _train_pytorch(model, train_dl, val_dl, device, input_key="global")
    return _metrics(y_val, probs)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 4 – CNN-LSTM
# ─────────────────────────────────────────────────────────────────────────────

class _CNNLSTMModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1,  32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(5),
            nn.Conv1d(32, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(5),
            nn.Conv1d(32, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(5),
        )
        # After 3×MaxPool1d(5): 2001 → 400 → 80 → 16
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2001, 1) → (batch, 1, 2001)
        cnn_out = self.cnn(x.transpose(1, 2))   # (batch, 32, 16)
        seq = cnn_out.transpose(1, 2)            # (batch, 16, 32)
        _, (hidden, _) = self.lstm(seq)          # hidden: (1, batch, 64)
        return self.head(hidden.squeeze(0))      # (batch, 1)


def _run_cnn_lstm(g_tr, l_tr, y_tr, g_val, l_val, y_val, device) -> dict:
    logger.info("Training CNN-LSTM …")
    train_ds = KeplerDataset(g_tr, l_tr, y_tr, augment=False)
    val_ds   = KeplerDataset(g_val, l_val, y_val, augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = _CNNLSTMModel()
    probs = _train_pytorch(model, train_dl, val_dl, device, input_key="global")
    return _metrics(y_val, probs)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_all_baselines(device: torch.device = None) -> pd.DataFrame:
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    logger.info("Loading fold 0 data …")
    g_tr, l_tr, y_tr, g_val, l_val, y_val = _load_fold0()
    logger.info(
        "  Train: %d samples | Val: %d samples | Device: %s",
        len(y_tr), len(y_val), device,
    )

    results = []

    m = _run_random_forest(g_tr, l_tr, y_tr, g_val, l_val, y_val)
    results.append({"Model": "Random Forest", **m})

    m = _run_vanilla_cnn(g_tr, l_tr, y_tr, g_val, l_val, y_val, device)
    results.append({"Model": "Vanilla 1D CNN", **m})

    m = _run_lstm(g_tr, l_tr, y_tr, g_val, l_val, y_val, device)
    results.append({"Model": "LSTM", **m})

    m = _run_cnn_lstm(g_tr, l_tr, y_tr, g_val, l_val, y_val, device)
    results.append({"Model": "CNN-LSTM", **m})

    df = pd.DataFrame(results, columns=["Model", "AUC-ROC", "F1", "Precision", "Recall", "AP"])

    os.makedirs(OUTPUT_RESULTS_PATH, exist_ok=True)
    csv_path = os.path.join(OUTPUT_RESULTS_PATH, "baseline_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Saved baseline results to %s", csv_path)

    print("\nBaseline Results")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    return df


if __name__ == "__main__":
    run_all_baselines()
