# Identification of Jets at the LHC - Deep Learning Spare Jet Classification

A machine learning mini-project (UCL - PHAS0057) classifying simulated LHC jets into five origin categories — Gluon, Quark, W boson, Z boson, Top quark — from calorimeter images and physics-derived jet features, using the public HLS4ML dataset.

**Main result:** a dense network (DNN) trained on 53 jet substructure features (115K parameters) outperformed a CNN trained on raw 100×100 calorimeter images (4.8M parameters) — 82.2% vs 73.3% validation accuracy, and a fivefold reduction in the prominent W/Z confusion.

---

## Motivation:

Particle jets — narrow collimated sprays of hadrons produced by high-momentum quarks, gluons, or boosted heavy particles — are central to LHC physics, from the original observation of the gluon to searches for rare Higgs decays. In experimental physics, the identification and classification of jets is one of the most important tasks as jet substructure has provided numerous alternative ways to to probe the Standard Model. The traditional methodology for jet classification involved relying on observables of the jets. However, with the advent of deep learning, a new approach for jet identification is available to us: translating the calorimeter energy deposits of a jet to a two-dimensional image and applying a convolutional neural network (CNN) directly to pixel-level data. 

This project aims to analyse the effectiveness of jet classification between five categories of particle - Gluon, Quark, W boson, Z boson and Top Quark - using a CNN only trained on sparse images of jet deposits, compared to jet identification using a DNN trained on 53 jet substructure observables. 

