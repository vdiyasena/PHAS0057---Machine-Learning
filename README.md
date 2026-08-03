# Identification of Jets at the LHC - Deep Learning Spare Jet Classification

A machine learning mini-project (UCL - PHAS0057) classifying simulated LHC jets into five origin categories — **Gluon, Quark, W boson, Z boson, Top quark** — from calorimeter images and physics-derived jet features, using the public **HLS4ML** dataset.

**Main result:** a dense network (DNN) trained on 53 jet substructure features (115K parameters) outperformed a CNN trained on raw 100×100 calorimeter images (4.8M parameters) — 82.2% vs 73.3% validation accuracy, and a fivefold reduction in the prominent W/Z confusion.


