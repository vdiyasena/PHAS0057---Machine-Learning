# Identification of Jets at the LHC - Deep Learning Spare Jet Classification

A machine learning mini-project (UCL - PHAS0057) classifying simulated LHC jets into five origin categories — Gluon, Quark, W boson, Z boson, Top quark — from calorimeter images and physics-derived jet features, using the public HLS4ML dataset.

**Main result:** a dense network (DNN) trained on 53 jet substructure features (115K parameters) outperformed a CNN trained on raw 100×100 calorimeter images (4.8M parameters) — 82.2% vs 73.3% validation accuracy, and a fivefold reduction in the prominent W/Z confusion.

---

## Motivation:

Particle jets — narrow collimated sprays of hadrons produced by high-momentum quarks, gluons, or boosted heavy particles — are central to LHC physics, from the original observation of the gluon to searches for rare Higgs decays. In experimental physics, the identification and classification of jets is one of the most important tasks as jet substructure has provided numerous alternative ways to to probe the Standard Model. The traditional methodology for jet classification involved relying on observables of the jets. However, with the advent of deep learning, a new approach for jet identification is available to us: translating the calorimeter energy deposits of a jet to a two-dimensional image and applying a convolutional neural network (CNN) directly to pixel-level data. 

## Data

- **Source:** [HLS4ML dataset](https://zenodo.org/doi/10.5281/zenodo.3602253) (Zenodo)
- **Input representations:**
  - 100×100 pixel calorimeter images (`jetImage`, and separately `jetImageECAL` / `jetImageHCAL`)
  - 53 jet substructure features (`j_pt`, `j_mass`, `j_multiplicity`, N-subjettiness variables, energy correlation functions, etc.)
- **Classes (one-hot → integer label):** `0=Gluon, 1=Quark, 2=W, 3=Z, 4=Top`
- **Preprocessing:** images log-normalised to [0, 1] (clipped to [0.01, 1000] before log); features standardised and NaN/Inf substructure values handled; data shuffled (`random_state=42`) before an 80/20 train/test split
- **Note on validation set:** the public HLS4ML release doesn't ship a separate held-out test set, so the validation files (7–9) were used only for early stopping, not weight updates, and then reused for evaluation metrics reported below

This project aims to analyse the effectiveness of jet classification between five categories of particle - Gluon, Quark, W boson, Z boson and Top Quark - using a CNN only trained on sparse images of jet deposits, compared to jet identification using a DNN trained on 53 jet substructure observables. 

