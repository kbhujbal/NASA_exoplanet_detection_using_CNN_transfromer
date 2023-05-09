import argparse
import json
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

from config import (
    DATA_RAW_PATH,
    LABEL_COLUMN,
    MC_DROPOUT_PASSES,
    OUTPUT_FIGURES_PATH,
    OUTPUT_MODELS_PATH,
    OUTPUT_RESULTS_PATH,
    PERIOD_COLUMN,
    EPOCH_COLUMN,
    UNCERTAINTY_HIGH,
    UNCERTAINTY_MEDIUM,
    N_FOLDS,
)
from src.preprocessing.cleaner import clean_light_curve
from src.preprocessing.detrending import detrend_flux
from src.preprocessing.phase_fold import phase_fold
from src.preprocessing.view_generator import generate_global_view, generate_local_view
from src.models.hybrid_model import HybridExoplanetModel
from src.uncertainty.mc_dropout import mc_dropout_predict, print_prediction_card
from src.explainability.attention_maps import extract_attention_weights, plot_attention_map
from src.explainability.gradcam import compute_gradcam, plot_gradcam

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CSV_FILENAME = "kepler_exoplanet_search_results.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict exoplanet candidacy for a single Kepler object."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--koi_id",
        type=str,
        help="Kepler Object ID (e.g. K00001.01 or KIC-11904151)",
    )
    group.add_argument(
        "--csv_row",
        type=int,
        help="Zero-based row index in the raw Kepler CSV",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Row lookup
# ─────────────────────────────────────────────────────────────────────────────

def _find_row(df: pd.DataFrame, koi_id: str | None, csv_row: int | None) -> pd.Series:
    if csv_row is not None:
        if csv_row < 0 or csv_row >= len(df):
            print(f"Error: --csv_row {csv_row} is out of range "
                  f"(CSV has {len(df)} data rows).")
            sys.exit(1)
        return df.iloc[csv_row]

    # Normalise the requested ID for flexible matching
    needle = koi_id.strip().lower().replace("kic-", "").replace("kic_", "")

    for col in ["kepoi_name", "kepid", "rowid"]:
        if col not in df.columns:
            continue
        haystack = df[col].astype(str).str.strip().str.lower()
        mask = (haystack == needle) | (haystack == koi_id.strip().lower())
        if mask.any():
            return df[mask].iloc[0]

    print(f"Error: KOI ID '{koi_id}' was not found in the dataset.")
    print("Tip: use --csv_row <index> to target a specific row by position.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Single-row preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_row(row: pd.Series, koi_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (global_view, local_view) each shape (L, 1), or raise ValueError."""
    period = row.get(PERIOD_COLUMN)
    epoch  = row.get(EPOCH_COLUMN)

    if pd.isna(period) or pd.isna(epoch) or float(period) <= 0:
        raise ValueError(
            f"KOI {koi_id}: missing or non-positive period/epoch "
            f"(period={period}, epoch={epoch})."
        )

    flux_cols = [c for c in row.index if c.startswith("flux_")]
    if not flux_cols:
        raise ValueError(
            f"KOI {koi_id}: no flux_N columns found in the CSV. "
            "The Kaggle KOI table does not include raw light-curve flux. "
            "Download the FITS light curves from MAST and add them as "
            "'flux_0', 'flux_1', … columns."
        )

    raw_flux = np.array([row[c] for c in flux_cols], dtype=float)
    time     = np.arange(len(raw_flux), dtype=float)

    cleaned = clean_light_curve(raw_flux)

    nan_mask = np.isnan(cleaned)
    if nan_mask.any():
        cleaned[nan_mask] = np.nanmedian(cleaned)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        detrended = detrend_flux(cleaned)

    folded_phase, folded_flux = phase_fold(time, detrended, float(period), float(epoch))

    global_view = generate_global_view(folded_phase, folded_flux)  # (2001, 1)
    local_view  = generate_local_view(folded_phase, folded_flux)   # (201, 1)

    return global_view, local_view


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_checkpoints(device: torch.device) -> list[HybridExoplanetModel]:
    paths = [
        os.path.join(OUTPUT_MODELS_PATH, f"fold_{i}_best.pt")
        for i in range(N_FOLDS)
    ]
    existing = [p for p in paths if os.path.exists(p)]

    if not existing:
        print("Error: No model checkpoints found in outputs/models/.")
        print("Run train.py first to train the models.")
        sys.exit(1)

    models = []
    for path in existing:
        model = HybridExoplanetModel().to(device)
        ckpt  = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)
        logger.info("Loaded checkpoint: %s", os.path.basename(path))

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble MC Dropout
# ─────────────────────────────────────────────────────────────────────────────

def _ensemble_mc_predict(
    models: list[HybridExoplanetModel],
    batch_dict: dict,
    device: torch.device,
) -> dict:
    """Pool all MC passes from every fold model, then compute combined statistics."""
    all_passes: list[float] = []

    for model in models:
        result = mc_dropout_predict(model, batch_dict, n_passes=MC_DROPOUT_PASSES, device=device)
        all_passes.extend(result["all_passes"])

    passes_arr  = np.array(all_passes)
    mean_prob   = float(passes_arr.mean())
    uncertainty = float(passes_arr.std())
    prediction  = "CONFIRMED" if mean_prob >= 0.5 else "FALSE POSITIVE"
    conf_pct    = mean_prob * 100 if prediction == "CONFIRMED" else (1.0 - mean_prob) * 100

    if uncertainty < UNCERTAINTY_HIGH:
        confidence_level = "HIGH"
    elif uncertainty < UNCERTAINTY_MEDIUM:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    return {
        "prediction":       prediction,
        "mean_probability": round(mean_prob, 6),
        "uncertainty":      round(uncertainty, 6),
        "confidence_pct":   round(conf_pct, 2),
        "confidence_level": confidence_level,
        "all_passes":       passes_arr.tolist(),
        "n_models_used":    len(models),
        "n_total_passes":   len(passes_arr),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(DATA_RAW_PATH, CSV_FILENAME)
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        print("Place kepler_exoplanet_search_results.csv in data/raw/ and retry.")
        sys.exit(1)

    logger.info("Loading %s …", csv_path)
    df = pd.read_csv(csv_path, comment="#")

    # ── Find row ──────────────────────────────────────────────────────────────
    row = _find_row(df, args.koi_id, args.csv_row)

    koi_id = str(row.get("kepoi_name", row.get("kepid", args.csv_row)))
    logger.info("Processing KOI: %s", koi_id)

    # ── Preprocessing ─────────────────────────────────────────────────────────
    try:
        global_view, local_view = _preprocess_row(row, koi_id)
    except ValueError as exc:
        print(f"\nPreprocessing failed: {exc}")
        sys.exit(1)

    batch_dict = {
        "global": torch.tensor(global_view[np.newaxis], dtype=torch.float32),
        "local":  torch.tensor(local_view[np.newaxis],  dtype=torch.float32),
    }

    # ── Load models ───────────────────────────────────────────────────────────
    models = _load_checkpoints(device)

    # ── Ensemble MC Dropout ───────────────────────────────────────────────────
    logger.info(
        "Running ensemble MC Dropout (%d models × %d passes) …",
        len(models), MC_DROPOUT_PASSES,
    )
    result = _ensemble_mc_predict(models, batch_dict, device)

    # ── Attention map ─────────────────────────────────────────────────────────
    xai_model = models[0]  # use fold-0 model for XAI
    os.makedirs(OUTPUT_FIGURES_PATH, exist_ok=True)

    attn_path = os.path.join(OUTPUT_FIGURES_PATH, f"attention_{koi_id}.png")
    try:
        attn_weights = extract_attention_weights(xai_model, batch_dict, device)
        plot_attention_map(global_view, attn_weights[0], koi_id,
                           result["prediction"], attn_path)
        logger.info("Attention map saved: %s", attn_path)
    except Exception as exc:
        logger.warning("Attention map generation failed: %s", exc)
        attn_path = None

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    gradcam_path = os.path.join(OUTPUT_FIGURES_PATH, f"gradcam_{koi_id}.png")
    try:
        cam = compute_gradcam(xai_model, batch_dict, target_class=1, device=device)
        plot_gradcam(global_view, cam[0], koi_id, result["prediction"], gradcam_path)
        logger.info("Grad-CAM saved: %s", gradcam_path)
    except Exception as exc:
        logger.warning("Grad-CAM generation failed: %s", exc)
        gradcam_path = None

    # ── Print prediction card ─────────────────────────────────────────────────
    print()
    print_prediction_card(koi_id, result)
    print(f"\n  MC passes   : {result['n_total_passes']} "
          f"({result['n_models_used']} models × {MC_DROPOUT_PASSES} passes)")
    print(f"  Mean P(planet): {result['mean_probability']:.4f}")
    if attn_path:
        print(f"  Attention map : {attn_path}")
    if gradcam_path:
        print(f"  Grad-CAM      : {gradcam_path}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    safe_id = koi_id.replace("/", "_").replace(" ", "_")
    os.makedirs(OUTPUT_RESULTS_PATH, exist_ok=True)
    json_path = os.path.join(OUTPUT_RESULTS_PATH, f"prediction_{safe_id}.json")

    output = {
        "koi_id":           koi_id,
        "prediction":       result["prediction"],
        "mean_probability": result["mean_probability"],
        "uncertainty":      result["uncertainty"],
        "confidence_pct":   result["confidence_pct"],
        "confidence_level": result["confidence_level"],
        "n_models_used":    result["n_models_used"],
        "n_total_passes":   result["n_total_passes"],
        "figures": {
            "attention_map": attn_path,
            "gradcam":       gradcam_path,
        },
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Result saved: %s", json_path)


if __name__ == "__main__":
    main()
