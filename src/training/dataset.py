import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import DATA_PROCESSED_PATH, RANDOM_SEED
from src.training.augmentation import augment_sample


class KeplerDataset(Dataset):
    def __init__(
        self,
        global_views: np.ndarray,
        local_views: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
    ) -> None:
        self.global_views = global_views.astype(np.float32)
        self.local_views = local_views.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        g = self.global_views[idx]   # (2001, 1)
        l = self.local_views[idx]    # (201, 1)

        if self.augment:
            g, l = augment_sample(g, l)

        return {
            "global": torch.tensor(g, dtype=torch.float32),
            "local": torch.tensor(l, dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def get_dataloaders(
    fold_idx: int,
    n_folds: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    global_views = np.load(os.path.join(DATA_PROCESSED_PATH, "global_views.npy"))  # (N, 2001, 1)
    local_views = np.load(os.path.join(DATA_PROCESSED_PATH, "local_views.npy"))    # (N, 201, 1)
    labels = np.load(os.path.join(DATA_PROCESSED_PATH, "labels.npy"))              # (N,)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    splits = list(skf.split(global_views, labels))
    train_idx, val_idx = splits[fold_idx]

    g_train = global_views[train_idx]   # (n_train, 2001, 1)
    l_train = local_views[train_idx]    # (n_train, 201, 1)
    y_train = labels[train_idx]

    # Flatten views for SMOTE, then reshape back
    n_train = g_train.shape[0]
    g_flat = g_train.reshape(n_train, -1)   # (n_train, 2001)
    l_flat = l_train.reshape(n_train, -1)   # (n_train, 201)
    combined = np.concatenate([g_flat, l_flat], axis=1)  # (n_train, 2202)

    smote = SMOTE(random_state=RANDOM_SEED)
    combined_resampled, y_resampled = smote.fit_resample(combined, y_train)

    g_resampled = combined_resampled[:, :2001].reshape(-1, 2001, 1)
    l_resampled = combined_resampled[:, 2001:].reshape(-1, 201, 1)

    g_val = global_views[val_idx]
    l_val = local_views[val_idx]
    y_val = labels[val_idx]

    train_dataset = KeplerDataset(g_resampled, l_resampled, y_resampled, augment=True)
    val_dataset = KeplerDataset(g_val, l_val, y_val, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader
