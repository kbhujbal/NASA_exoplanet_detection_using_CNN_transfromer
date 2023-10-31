import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import torch

from config import (
    DATA_PROCESSED_PATH,
    DATA_RAW_PATH,
    OUTPUT_RESULTS_PATH,
    RANDOM_SEED,
)
from src.preprocessing.view_generator import run_full_preprocessing
from src.training.trainer import run_kfold_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Hybrid CNN-Transformer for exoplanet detection")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on: 'cuda', 'mps', or 'cpu'. Auto-detected if omitted.",
    )
    return parser.parse_args()


def resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    set_seeds(RANDOM_SEED)

    device = resolve_device(args.device)
    logger.info("Using device: %s", device)

    processed_flag = os.path.join(DATA_PROCESSED_PATH, "global_views.npy")
    if os.path.exists(processed_flag):
        logger.info("Using cached processed data from %s", DATA_PROCESSED_PATH)
    else:
        logger.info("Processed data not found — running preprocessing …")
        csv_path = os.path.join(DATA_RAW_PATH, "kepler_exoplanet_search_results.csv")
        run_full_preprocessing(csv_path)

    all_histories = run_kfold_training(device)

    os.makedirs(OUTPUT_RESULTS_PATH, exist_ok=True)
    history_path = os.path.join(OUTPUT_RESULTS_PATH, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(all_histories, f, indent=2)
    logger.info("Training histories saved to %s", history_path)

    print("\nTraining complete. Models saved to outputs/models/")


if __name__ == "__main__":
    main()
