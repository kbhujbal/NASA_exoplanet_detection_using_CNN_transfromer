# Exoplanet Transit Detection using Hybrid CNN-Transformer with Explainable AI

A deep learning pipeline for automated classification of Kepler Objects of Interest as confirmed exoplanets or false positives, combining multi-scale convolutional feature extraction with Transformer-based sequence modelling and integrated explainability.

---

## Abstract

The automated vetting of exoplanet transit signals from photometric survey data presents a significant classification challenge due to the scale of modern datasets and the morphological similarity between genuine planetary transits and astrophysical false positives such as eclipsing binaries and background contaminants. We present a hybrid architecture that processes phase-folded Kepler light curves through three parallel branches — a global-view CNN, a local-view CNN, and a Transformer encoder — whose outputs are fused and classified by a shared MLP head trained with Focal Loss to address severe class imbalance. Monte Carlo Dropout provides calibrated epistemic uncertainty estimates, enabling the system to flag ambiguous candidates for human review rather than issuing low-confidence classifications silently. Gradient-weighted Class Activation Mapping, Transformer attention analysis, and SHAP values supply complementary post-hoc explanations that allow astronomers to audit the evidence underlying each prediction. The full pipeline is evaluated under 5-fold stratified cross-validation and benchmarked against four baselines spanning classical ensemble methods and recurrent architectures.

---

## Architecture

```
 Global View (2001 x 1)        Local View (201 x 1)          Global View (2001 x 1)
         |                             |                               |
   +-----+------+               +-----+------+               +--------+--------+
   |  CNN Global |               |  CNN Local  |               | Transformer     |
   |             |               |             |               | Branch          |
   |  Conv1d x4  |               |  Conv1d x3  |               |                 |
   |  [16,32,64  |               |  [16,32,64] |               | Linear(1->64)   |
   |   ,128]     |               |  kernel=3   |               | Sinusoidal PE   |
   |  kernel=5   |               |  MaxPool x3 |               | 4x Encoder      |
   |  MaxPool x4 |               |  FC -> 256  |               |   d=64, heads=8 |
   |  FC -> 512  |               |             |               | GlobalAvgPool   |
   +-----+-------+               +-----+-------+               | FC -> 256       |
         | (512)                       | (256)                 +--------+--------+
         +-------------------+---------+                                | (256)
                             |<---------------------------------------+
                             |   Concatenate -> (1024,)
                       +-----+------+
                       |    MLP     |
                       |    Head    |
                       |            |
                       | Linear(1024->512)  |
                       | BatchNorm + Dropout(0.5) |
                       | Linear(512->256)   |
                       | Dropout(0.3)       |
                       | Linear(256->1)     |
                       | Sigmoid            |
                       +-----+------+
                             |
                       P(planet) in [0, 1]
```

**Training components:**
- Loss: Focal Loss (alpha=0.25, gamma=2.0)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- Imbalance handling: SMOTE on training folds
- Augmentation: circular time shift, Gaussian noise, flux scaling
- Validation: 5-fold stratified cross-validation with early stopping (patience=10)

---

## Project Structure

```
NASA_exoplanet_detection_using_CNN_transfromer/
|
|-- config.py                        # All hyperparameters and path constants
|-- train.py                         # K-fold training entry point
|-- evaluate.py                      # Ensemble evaluation and plot generation
|-- predict.py                       # Single-KOI inference CLI
|-- setup.py                         # Creates data/ and outputs/ directory tree
|-- requirements.txt                 # Pinned Python dependencies
|
|-- data/
|   |-- raw/                         # Place kepler_exoplanet_search_results.csv here
|   |-- processed/                   # Auto-generated .npy files after preprocessing
|   +-- augmented/                   # Reserved for future use
|
|-- notebooks/
|   +-- full_pipeline.ipynb          # End-to-end walkthrough (supplementary material)
|
|-- outputs/
|   |-- models/                      # Saved fold checkpoints (fold_0_best.pt, ...)
|   |-- figures/                     # All plots (ROC, attention maps, Grad-CAM, SHAP)
|   +-- results/                     # CSVs and JSON (training_history, baselines, etc.)
|
+-- src/
    |-- preprocessing/
    |   |-- cleaner.py               # Outlier removal and NaN gap handling
    |   |-- detrending.py            # Savitzky-Golay baseline removal
    |   |-- phase_fold.py            # Period-epoch phase folding
    |   +-- view_generator.py        # Global/local view binning and full pipeline runner
    |
    |-- models/
    |   |-- cnn_global.py            # 4-block 1D CNN for global view (output: 512)
    |   |-- cnn_local.py             # 3-block 1D CNN for local view (output: 256)
    |   |-- transformer.py           # Sinusoidal PE + 4-layer Transformer encoder (output: 256)
    |   +-- hybrid_model.py          # Fusion model with MC Dropout support
    |
    |-- training/
    |   |-- augmentation.py          # Time shift, noise, and flux scaling augmentations
    |   |-- dataset.py               # KeplerDataset and get_dataloaders (SMOTE + KFold)
    |   |-- losses.py                # Focal Loss implementation
    |   +-- trainer.py               # Trainer class and run_kfold_training entry point
    |
    |-- evaluation/
    |   +-- metrics.py               # compute_all_metrics and ensemble_predict
    |
    |-- uncertainty/
    |   +-- mc_dropout.py            # MC Dropout inference and prediction card printer
    |
    |-- explainability/
    |   |-- attention_maps.py        # Transformer attention weight extraction and plotting
    |   |-- gradcam.py               # Grad-CAM on last CNN conv layer
    |   +-- shap_analysis.py         # DeepExplainer wrapper, summary and waterfall plots
    |
    |-- baselines/
    |   +-- baselines.py             # Random Forest, Vanilla CNN, LSTM, CNN-LSTM
    |
    +-- visualization/
        +-- plots.py                 # ROC, PR, confusion matrix, training curves, bar chart
```

---

## Installation

**Requirements:** Python 3.10+, pip

```bash
git clone https://github.com/<your-username>/NASA_exoplanet_detection_using_CNN_transfromer.git
cd NASA_exoplanet_detection_using_CNN_transfromer

pip install -r requirements.txt

# Create the directory structure
python setup.py
```

To use GPU acceleration, ensure a CUDA-compatible PyTorch build is installed. The code auto-detects `cuda > mps > cpu` at runtime.

---

## Dataset Setup

1. Download the Kepler Exoplanet Search Results dataset from Kaggle:
   [https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results)

2. Place the CSV file in the raw data directory:
   ```
   data/raw/kepler_exoplanet_search_results.csv
   ```

3. Preprocessing runs automatically on the first call to `train.py`. To run it manually:
   ```python
   from src.preprocessing.view_generator import run_full_preprocessing
   run_full_preprocessing("data/raw/kepler_exoplanet_search_results.csv")
   ```
   This generates `global_views.npy`, `local_views.npy`, `labels.npy`, and `koi_ids.npy` in `data/processed/`.

**Note:** The Kaggle KOI table contains derived orbital parameters but not raw photometric flux time-series. To use the preprocessing pipeline on actual light curves, download FITS files from the [MAST archive](https://mast.stsci.edu) and add flux columns (`flux_0`, `flux_1`, ...) to the CSV, or use the `lightkurve` package to fetch and inject them programmatically.

---

## Usage

### Training

Runs 5-fold stratified cross-validation. Preprocesses data automatically if `data/processed/` is empty.

```bash
python train.py
# or specify device explicitly:
python train.py --device cuda
python train.py --device cpu
```

Checkpoints are saved to `outputs/models/fold_{0..4}_best.pt`. Training history is saved to `outputs/results/training_history.json`.

### Evaluation

Loads all fold checkpoints, runs ensemble inference, generates all figures, and prints the final comparison table.

```bash
python evaluate.py
```

Outputs:
- `outputs/figures/` — 20 plots including ROC curves, attention maps, Grad-CAM, SHAP summary
- `outputs/results/baseline_results.csv` — baseline metrics
- `outputs/results/final_comparison.csv` — full model comparison table

### Single-KOI Inference

```bash
# By Kepler Object ID
python predict.py --koi_id K00001.01

# By row index in the CSV
python predict.py --csv_row 42
```

Outputs a formatted prediction card, saves attention map and Grad-CAM to `outputs/figures/`, and writes a JSON result to `outputs/results/prediction_<koi_id>.json`.

### Notebook

```bash
jupyter notebook notebooks/full_pipeline.ipynb
```

The notebook is a self-contained, sequentially runnable walkthrough covering dataset exploration, preprocessing visualisation, model architecture, training results, XAI analysis, uncertainty quantification, and live inference. All cells degrade gracefully when trained checkpoints are not yet available.

---

## Results

All metrics are reported on the held-out validation fold (fold 0) unless otherwise noted. Fill in the table after running `python evaluate.py`.

| Model | AUC-ROC | F1 | Precision | Recall | Avg. Precision |
|---|---|---|---|---|---|
| Hybrid CNN-Transformer (Ours) | — | — | — | — | — |
| Random Forest (200 trees) | — | — | — | — | — |
| Vanilla 1D CNN | — | — | — | — | — |
| Bidirectional LSTM | — | — | — | — | — |
| CNN-LSTM | — | — | — | — | — |

**Cross-validation AUC (5 folds):** — +/- — *(populate from `outputs/results/training_history.json`)*

---

## Key Contributions

- **Hybrid multi-branch architecture**: A novel fusion of a dual-scale CNN (global and local phase-folded views) with a Transformer encoder that captures long-range temporal dependencies across the full orbital phase, addressing the complementary strengths of local transit morphology and global light-curve context.

- **Uncertainty-aware inference**: Monte Carlo Dropout at test time produces calibrated epistemic uncertainty estimates that partition predictions into HIGH, MEDIUM, and LOW confidence tiers, enabling principled deferral to astronomer review for ambiguous candidates.

- **Integrated XAI suite**: Three complementary interpretability methods — Grad-CAM on the final convolutional layer, per-layer Transformer attention weight aggregation, and SHAP DeepExplainer values — provide spatially resolved, human-auditable evidence for every classification decision.

- **End-to-end reproducible pipeline**: A single command (`python train.py`) executes preprocessing, SMOTE oversampling, data augmentation, 5-fold cross-validation, early stopping, and checkpoint saving; `evaluate.py` then generates all publication-quality figures and the full baseline comparison automatically.

---

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@misc{exoplanet_hybrid_cnn_transformer,
  author    = {[Author Names]},
  title     = {Exoplanet Transit Detection using Hybrid CNN-Transformer
               with Explainable AI},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/<your-username>/NASA_exoplanet_detection_using_CNN_transfromer},
  note      = {Software pipeline for automated KOI classification
               from Kepler photometry}
}
```

---

## License

This project is released under the MIT License.

```
MIT License

Copyright (c) 2025 Kunal Bhujbal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
