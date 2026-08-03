# Identification of Jets at the LHC - Deep Learning Spare Jet Classification

A machine learning mini-project (UCL - PHAS0057) classifying simulated LHC jets into five origin categories — Gluon, Quark, W boson, Z boson, Top quark — from calorimeter images and physics-derived jet features, using the public HLS4ML dataset.

**Main result:** a dense network (DNN) trained on 53 jet substructure features (115K parameters) outperformed a CNN trained on raw 100×100 calorimeter images (4.8M parameters) — 82.2% vs 73.3% validation accuracy, and a fivefold reduction in the prominent W/Z confusion.

---

## Motivation:

Particle jets — narrow collimated sprays of hadrons produced by high-momentum quarks, gluons, or boosted heavy particles — are central to LHC physics, from the original observation of the gluon to searches for rare Higgs decays. In experimental physics, the identification and classification of jets is one of the most important tasks as jet substructure has provided numerous alternative ways to to probe the Standard Model. The traditional methodology for jet classification involved relying on observables of the jets. However, with the advent of deep learning, a new approach for jet identification is available to us: translating the calorimeter energy deposits of a jet to a two-dimensional image and applying a convolutional neural network (CNN) directly to pixel-level data. 

---

## Data

- **Source:** [HLS4ML dataset](https://zenodo.org/doi/10.5281/zenodo.3602253) (Zenodo)
- **Input representations:**
  - 100×100 pixel calorimeter images (`jetImage`, and separately `jetImageECAL` / `jetImageHCAL`)
  - 53 jet substructure features (`j_pt`, `j_mass`, `j_multiplicity`, N-subjettiness variables, energy correlation functions, etc.)
- **Classes (one-hot → integer label):** `0=Gluon, 1=Quark, 2=W, 3=Z, 4=Top`
- **Preprocessing:** images log-normalised to [0, 1] (clipped to [0.01, 1000] before log); features standardised and NaN/Inf substructure values handled; data shuffled (`random_state=42`) before an 80/20 train/test split
- **Note on validation set:** the public HLS4ML release doesn't ship a separate held-out test set, so the validation files (7–9) were used only for early stopping, not weight updates, and then reused for evaluation metrics reported below

This project aims to analyse the effectiveness of jet classification between five categories of particle - Gluon, Quark, W boson, Z boson and Top Quark - using a CNN only trained on sparse images of jet deposits, compared to jet identification using a DNN trained on 53 jet substructure observables. 

---

## Models

### 1. Image CNN (`model_v1`) — baseline
Trained on the single-channel combined calorimeter image.

- 3× `Conv2D` blocks (32 → 64 → 128 filters, 3×3 kernels, LeakyReLU, MaxPooling, L2 regularisation)
- `Dense(256)` → Dropout(0.3) → `Dense(5)` output (softmax via logits + `SparseCategoricalCrossentropy`)
- Optimizer: Adam · Early stopping on `val_loss` (patience 10) · up to 200 epochs

### 2. Features DNN — extension
Trained on the 53 scalar jet observables only (no image input), architecture-matched in depth to `model_v1` for a fair comparison.

- 3× `Dense` blocks (256 → 256 → 128, each with BatchNorm, LeakyReLU, Dropout 0.3, L2 regularisation)
- `Dense(5)` output
- Optimizer: AdamW · same early-stopping regime

### 3. `model_v2` (attempted, not adopted)
A deeper/wider CNN redesign (2×2 kernels, more filters, a fourth conv block, BatchNorm) motivated by the diagnostics below. It **overfit** — worse validation accuracy despite marginally better training accuracy — and was abandoned (and thus not mentioned in final report).

---

## Results

### Headline metrics

| Model | Input | Val/Test Accuracy | Macro AUC | Params |
|---|---|---:|---:|---:|
| Image CNN (`model_v1`) | 100×100 calorimeter image | 0.733 | 0.928 | 4,812,805 |
| Features DNN | 53 jet observables | 0.822 | 0.962 | 115,717 |

### Image CNN — per-class performance (test set, n=16,000)

| Class | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|
| Gluon | 0.722 | 0.676 | 0.698 | 0.922 |
| Quark | 0.662 | 0.697 | 0.679 | 0.906 |
| W | 0.710 | 0.802 | 0.753 | 0.937 |
| Z | 0.771 | 0.695 | 0.731 | 0.923 |
| Top | 0.808 | 0.790 | 0.799 | 0.951 |

