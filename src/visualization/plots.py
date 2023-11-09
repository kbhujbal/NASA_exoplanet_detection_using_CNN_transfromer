import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import OUTPUT_FIGURES_PATH

_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _save(fig, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(models_dict: dict, save_path: str) -> None:
    """models_dict = {"Model Name": (y_true, y_prob), ...}"""
    fig, ax = plt.subplots(figsize=(8, 7))

    for i, (name, (y_true, y_prob)) in enumerate(models_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        lw = 2.5 if i == 0 else 1.5
        ax.plot(fpr, tpr, color=_COLORS[i % len(_COLORS)],
                linewidth=lw, label=f"{name}  (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, save_path)


def plot_pr_curves(models_dict: dict, save_path: str) -> None:
    """models_dict = {"Model Name": (y_true, y_prob), ...}"""
    fig, ax = plt.subplots(figsize=(8, 7))

    for i, (name, (y_true, y_prob)) in enumerate(models_dict.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        lw = 2.5 if i == 0 else 1.5
        ax.plot(recall, precision, color=_COLORS[i % len(_COLORS)],
                linewidth=lw, label=f"{name}  (AP = {ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — All Models", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    _save(fig, save_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
) -> None:
    from sklearn.metrics import confusion_matrix as sk_cm
    labels = ["FALSE POSITIVE", "CONFIRMED"]
    cm = sk_cm(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — Hybrid CNN-Transformer Ensemble", fontsize=13)
    _save(fig, save_path)


def plot_training_curves(histories: list, save_path: str) -> None:
    """histories = list of per-fold history dicts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for fold_idx, hist in enumerate(histories):
        color = _COLORS[fold_idx % len(_COLORS)]
        epochs = range(1, len(hist["train_loss"]) + 1)

        axes[0].plot(epochs, hist["train_loss"], color=color,
                     linestyle="--", alpha=0.7, linewidth=1.2)
        axes[0].plot(epochs, hist["val_loss"], color=color,
                     linestyle="-", linewidth=1.5, label=f"Fold {fold_idx}")

        axes[1].plot(epochs, hist["val_auc"], color=color,
                     linewidth=1.5, label=f"Fold {fold_idx}")

    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Focal Loss", fontsize=11)
    axes[0].set_title("Train (dashed) / Val (solid) Loss per Fold", fontsize=12)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("AUC-ROC", fontsize=11)
    axes[1].set_title("Validation AUC per Fold", fontsize=12)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Training Curves — All Folds", fontsize=14, y=1.02)
    _save(fig, save_path)


def plot_model_comparison(results_df: pd.DataFrame, save_path: str) -> None:
    """results_df must have columns: Model, AUC-ROC, F1, AP."""
    metrics = ["AUC-ROC", "F1", "AP"]
    n_models = len(results_df)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(9, n_models * 2), 6))

    for i, metric in enumerate(metrics):
        vals = results_df[metric].values
        bars = ax.bar(x + i * width, vals, width, label=metric,
                      color=_COLORS[i], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels(results_df["Model"].values, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title("Model Comparison — AUC-ROC, F1, Average Precision", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, save_path)


def plot_uncertainty_distribution(
    uncertainty_array: np.ndarray,
    predictions: np.ndarray,
    save_path: str,
) -> None:
    """
    uncertainty_array : (N,) MC-dropout std values
    predictions       : (N,) boolean — True = model prediction was correct
    """
    correct   = uncertainty_array[predictions]
    incorrect = uncertainty_array[~predictions]

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, uncertainty_array.max() + 0.01, 40)

    ax.hist(correct,   bins=bins, alpha=0.7, color="steelblue",
            label=f"Correct  (n={len(correct)})",   edgecolor="white")
    ax.hist(incorrect, bins=bins, alpha=0.7, color="tomato",
            label=f"Incorrect (n={len(incorrect)})", edgecolor="white")

    ax.axvline(0.05, color="orange", linestyle="--", linewidth=1.2, label="HIGH threshold (0.05)")
    ax.axvline(0.15, color="red",    linestyle="--", linewidth=1.2, label="LOW  threshold (0.15)")

    ax.set_xlabel("MC Dropout Uncertainty (σ)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Uncertainty Distribution — Correct vs Incorrect Predictions", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    _save(fig, save_path)
