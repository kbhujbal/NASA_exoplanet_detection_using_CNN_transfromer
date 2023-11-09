import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    DATA_PROCESSED_PATH,
    N_FOLDS,
    OUTPUT_FIGURES_PATH,
    OUTPUT_MODELS_PATH,
    OUTPUT_RESULTS_PATH,
    RANDOM_SEED,
)
from src.baselines.baselines import (
    _CNNLSTMModel,
    _LSTMModel,
    _VanillaCNN,
    _load_fold0,
    _train_pytorch,
)
from src.evaluation.metrics import compute_all_metrics, ensemble_predict
from src.explainability.attention_maps import extract_attention_weights, plot_attention_map
from src.explainability.gradcam import compute_gradcam, plot_gradcam
from src.explainability.shap_analysis import (
    compute_shap_values,
    plot_shap_summary,
    plot_shap_waterfall,
)
from src.models.hybrid_model import HybridExoplanetModel
from src.training.dataset import KeplerDataset
from src.uncertainty.mc_dropout import batch_mc_predict, print_prediction_card
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_pr_curves,
    plot_roc_curves,
    plot_training_curves,
    plot_uncertainty_distribution,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_val_data():
    """Return val split (fold 0) arrays and koi_ids."""
    global_views = np.load(os.path.join(DATA_PROCESSED_PATH, "global_views.npy"))
    local_views  = np.load(os.path.join(DATA_PROCESSED_PATH, "local_views.npy"))
    labels       = np.load(os.path.join(DATA_PROCESSED_PATH, "labels.npy"))
    koi_ids      = np.load(os.path.join(DATA_PROCESSED_PATH, "koi_ids.npy"), allow_pickle=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    train_idx, val_idx = list(skf.split(global_views, labels))[0]

    return (
        global_views[train_idx], local_views[train_idx], labels[train_idx],
        global_views[val_idx],   local_views[val_idx],   labels[val_idx],
        koi_ids[val_idx],
    )


def _make_val_loader(g_val, l_val, y_val) -> DataLoader:
    ds = KeplerDataset(g_val, l_val, y_val, augment=False)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


def _load_single_model(checkpoint_path: str, device: torch.device) -> HybridExoplanetModel:
    model = HybridExoplanetModel().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _get_baseline_predictions(device: torch.device, g_tr, l_tr, y_tr, g_val, l_val, y_val):
    """Retrain all 4 baseline models and return per-sample val probabilities."""
    logger.info("Retraining baselines for ROC/PR curves …")

    # Random Forest — flat features
    X_tr  = np.concatenate([g_tr.reshape(len(g_tr), -1), l_tr.reshape(len(l_tr), -1)], axis=1)
    X_val = np.concatenate([g_val.reshape(len(g_val), -1), l_val.reshape(len(l_val), -1)], axis=1)
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_probs = rf.predict_proba(X_val)[:, 1]

    # Shared DataLoaders for PyTorch baselines
    train_ds = KeplerDataset(g_tr, l_tr, y_tr, augment=False)
    val_ds   = KeplerDataset(g_val, l_val, y_val, augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    cnn_probs      = _train_pytorch(_VanillaCNN(),   train_dl, val_dl, device, "global")
    lstm_probs     = _train_pytorch(_LSTMModel(),    train_dl, val_dl, device, "global")
    cnn_lstm_probs = _train_pytorch(_CNNLSTMModel(), train_dl, val_dl, device, "global")

    return {
        "Random Forest":  rf_probs,
        "Vanilla 1D CNN": cnn_probs,
        "LSTM":           lstm_probs,
        "CNN-LSTM":       cnn_lstm_probs,
    }


def _single_sample_dict(g_val, l_val, idx: int, device: torch.device) -> dict:
    return {
        "global": torch.tensor(g_val[idx : idx + 1], dtype=torch.float32).to(device),
        "local":  torch.tensor(l_val[idx : idx + 1], dtype=torch.float32).to(device),
    }


def _fig_path(filename: str) -> str:
    os.makedirs(OUTPUT_FIGURES_PATH, exist_ok=True)
    return os.path.join(OUTPUT_FIGURES_PATH, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = _resolve_device()
    logger.info("Device: %s", device)

    # ── 1. Load fold-0 val data ──────────────────────────────────────────────
    logger.info("Loading fold-0 val data …")
    g_tr, l_tr, y_tr, g_val, l_val, y_val, koi_ids_val = _load_val_data()
    val_loader = _make_val_loader(g_val, l_val, y_val)
    logger.info("Val set: %d samples", len(y_val))

    # ── 2. Ensemble inference ────────────────────────────────────────────────
    model_paths = [
        os.path.join(OUTPUT_MODELS_PATH, f"fold_{i}_best.pt") for i in range(N_FOLDS)
    ]
    existing_paths = [p for p in model_paths if os.path.exists(p)]

    if existing_paths:
        logger.info("Running ensemble inference on %d checkpoints …", len(existing_paths))
        ens_probs = ensemble_predict(existing_paths, val_loader, device)
        ens_metrics = compute_all_metrics(y_val, ens_probs)
        logger.info("Ensemble AUC=%.4f  F1=%.4f", ens_metrics["auc_roc"], ens_metrics["f1"])
    else:
        logger.warning("No model checkpoints found — skipping ensemble evaluation.")
        ens_probs = None

    # ── 3. Load one model for XAI ────────────────────────────────────────────
    xai_model = None
    if existing_paths:
        xai_model = _load_single_model(existing_paths[0], device)

    # ── 4. MC Dropout uncertainty ────────────────────────────────────────────
    mc_df = None
    if xai_model is not None:
        logger.info("Running MC Dropout …")
        mc_df = batch_mc_predict(xai_model, val_loader, device)
        mc_df["true_label"] = y_val

    # ── 5. Baseline predictions ──────────────────────────────────────────────
    baseline_preds = _get_baseline_predictions(device, g_tr, l_tr, y_tr, g_val, l_val, y_val)

    # ── 6. Build models_dict for curves ─────────────────────────────────────
    models_dict = {}
    if ens_probs is not None:
        models_dict["Hybrid CNN-Transformer (Ours)"] = (y_val, ens_probs)
    for name, probs in baseline_preds.items():
        models_dict[name] = (y_val, probs)

    # ── 7. Plot 1 — ROC curves ───────────────────────────────────────────────
    logger.info("Plotting ROC curves …")
    plot_roc_curves(models_dict, _fig_path("roc_curves_all_models.png"))

    # ── 8. Plot 2 — PR curves ────────────────────────────────────────────────
    logger.info("Plotting PR curves …")
    plot_pr_curves(models_dict, _fig_path("pr_curves_all_models.png"))

    # ── 9. Plot 3 — Confusion matrix ─────────────────────────────────────────
    if ens_probs is not None:
        logger.info("Plotting confusion matrix …")
        ens_preds = (ens_probs >= 0.5).astype(int)
        plot_confusion_matrix(y_val, ens_preds, _fig_path("confusion_matrix_ours.png"))

    # ── 10. Plot 4 — Training curves ─────────────────────────────────────────
    history_path = os.path.join(OUTPUT_RESULTS_PATH, "training_history.json")
    if os.path.exists(history_path):
        logger.info("Plotting training curves …")
        with open(history_path) as f:
            histories = json.load(f)
        plot_training_curves(histories, _fig_path("training_curves_all_folds.png"))
    else:
        logger.warning("training_history.json not found, skipping training curves.")

    # ── 11. Plot 5 — Model comparison bar chart ───────────────────────────────
    logger.info("Plotting model comparison …")
    compare_rows = []
    if ens_probs is not None:
        m = ens_metrics
        compare_rows.append({
            "Model": "Hybrid CNN-Transformer",
            "AUC-ROC": round(m["auc_roc"], 4),
            "F1":      round(m["f1"], 4),
            "AP":      round(m["average_precision"], 4),
        })
    for name, probs in baseline_preds.items():
        m = compute_all_metrics(y_val, probs)
        compare_rows.append({
            "Model":   name,
            "AUC-ROC": round(m["auc_roc"], 4),
            "F1":      round(m["f1"], 4),
            "AP":      round(m["average_precision"], 4),
        })
    compare_df = pd.DataFrame(compare_rows)
    plot_model_comparison(compare_df, _fig_path("model_comparison_bar.png"))

    # ── 12. Plot 6 — Uncertainty distribution ────────────────────────────────
    if mc_df is not None:
        logger.info("Plotting uncertainty distribution …")
        uncertainties = mc_df["uncertainty"].values
        mc_pred_numeric = mc_df["prediction"].map({"CONFIRMED": 1, "FALSE POSITIVE": 0}).values
        correct_mask = mc_pred_numeric == y_val
        plot_uncertainty_distribution(
            uncertainties, correct_mask, _fig_path("uncertainty_distribution.png")
        )

    # ── 13-15. Attention maps + Grad-CAM for special samples ─────────────────
    if xai_model is not None and mc_df is not None:
        logger.info("Computing attention maps and Grad-CAM for selected samples …")

        # Identify sample groups
        confirmed_mask = mc_df["prediction"] == "CONFIRMED"
        fp_mask        = mc_df["prediction"] == "FALSE POSITIVE"

        top_planet_idx = (
            mc_df[confirmed_mask]["confidence_pct"].nlargest(3).index.tolist()
        )
        top_fp_idx = (
            mc_df[fp_mask]["confidence_pct"].nlargest(3).index.tolist()
        )
        top_uncertain_idx = (
            mc_df["uncertainty"].nlargest(3).index.tolist()
        )

        sample_groups = {
            "planet":    top_planet_idx,
            "fp":        top_fp_idx,
            "uncertain": top_uncertain_idx,
        }

        for group_name, indices in sample_groups.items():
            for rank, sample_idx in enumerate(indices, start=1):
                koi_id     = str(koi_ids_val[sample_idx])
                prediction = mc_df.loc[sample_idx, "prediction"]
                batch_dict = _single_sample_dict(g_val, l_val, sample_idx, device)

                # Attention map
                try:
                    attn = extract_attention_weights(xai_model, batch_dict, device)
                    attn_path = _fig_path(f"attention_map_{group_name}_{rank}.png")
                    plot_attention_map(g_val[sample_idx], attn[0], koi_id, prediction, attn_path)
                    logger.info("  Saved attention map: %s", attn_path)
                except Exception as exc:
                    logger.warning("  Attention map failed for %s: %s", koi_id, exc)

                # Grad-CAM
                try:
                    cam_path = _fig_path(f"gradcam_{group_name}_{rank}.png")
                    cam = compute_gradcam(xai_model, batch_dict, target_class=1, device=device)
                    plot_gradcam(g_val[sample_idx], cam[0], koi_id, prediction, cam_path)
                    logger.info("  Saved Grad-CAM: %s", cam_path)
                except Exception as exc:
                    logger.warning("  Grad-CAM failed for %s: %s", koi_id, exc)

    # ── 16. SHAP (plots 19-20) ────────────────────────────────────────────────
    if xai_model is not None:
        logger.info("Computing SHAP values (this may take a few minutes) …")
        try:
            train_ds    = KeplerDataset(g_tr, l_tr, y_tr, augment=False)
            val_ds      = KeplerDataset(g_val, l_val, y_val, augment=False)
            bg_loader   = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
            shap_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

            shap_dict = compute_shap_values(
                xai_model, bg_loader, shap_loader,
                n_background=100, n_test=200, device=device,
            )

            # Plot 19 — SHAP summary
            plot_shap_summary(shap_dict, _fig_path("shap_summary.png"))
            logger.info("  Saved shap_summary.png")

            # Plot 20 — SHAP waterfall for best-confidence planet prediction
            if mc_df is not None:
                confirmed_rows = mc_df[mc_df["prediction"] == "CONFIRMED"]
                if not confirmed_rows.empty:
                    best_planet_df_idx = confirmed_rows["confidence_pct"].idxmax()
                    # Map from mc_df row index to shap test array index
                    shap_sample_idx = min(best_planet_df_idx, shap_dict["shap_values"].shape[0] - 1)
                    koi_id     = str(koi_ids_val[best_planet_df_idx])
                    prediction = "CONFIRMED"
                    plot_shap_waterfall(
                        shap_dict, shap_sample_idx, koi_id, prediction,
                        _fig_path("shap_waterfall_best_planet.png"),
                    )
                    logger.info("  Saved shap_waterfall_best_planet.png")
        except Exception as exc:
            logger.warning("SHAP computation failed: %s", exc)

    # ── 17. Print prediction cards for top 3 confirmed ────────────────────────
    if mc_df is not None:
        confirmed_rows = mc_df[mc_df["prediction"] == "CONFIRMED"]
        top3 = confirmed_rows.nlargest(3, "confidence_pct")
        print("\n=== Top 3 Most Confident CONFIRMED Predictions ===")
        for idx in top3.index:
            row = mc_df.loc[idx]
            result = {
                "prediction":     row["prediction"],
                "confidence_pct": row["confidence_pct"],
                "uncertainty":    row["uncertainty"],
                "confidence_level": row["confidence_level"],
            }
            print_prediction_card(str(koi_ids_val[idx]), result)

    # ── 18. Final comparison table ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Final Model Comparison")
    print("=" * 70)

    full_rows = []
    if ens_probs is not None:
        m = compute_all_metrics(y_val, ens_probs)
        full_rows.append({
            "Model":     "Hybrid CNN-Transformer (Ours)",
            "AUC-ROC":   f"{m['auc_roc']:.4f}",
            "F1":        f"{m['f1']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall":    f"{m['recall']:.4f}",
            "AP":        f"{m['average_precision']:.4f}",
        })
    for name, probs in baseline_preds.items():
        m = compute_all_metrics(y_val, probs)
        full_rows.append({
            "Model":     name,
            "AUC-ROC":   f"{m['auc_roc']:.4f}",
            "F1":        f"{m['f1']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall":    f"{m['recall']:.4f}",
            "AP":        f"{m['average_precision']:.4f}",
        })

    final_df = pd.DataFrame(full_rows)
    print(final_df.to_string(index=False))
    print("=" * 70)

    # Save final results table
    os.makedirs(OUTPUT_RESULTS_PATH, exist_ok=True)
    final_df.to_csv(os.path.join(OUTPUT_RESULTS_PATH, "final_comparison.csv"), index=False)
    logger.info("Saved final_comparison.csv")

    print(f"\nAll figures saved to {OUTPUT_FIGURES_PATH}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
