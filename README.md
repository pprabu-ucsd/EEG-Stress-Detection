# EEG Stress Detection

**Pranav Prabu, Selina Wu, Calwin Li**  
COGS 189: Brain Computer Interfaces — Winter 2026, UC San Diego
[Link to Paper](https://docs.google.com/document/d/1raws2RGpRGAk9-RfQMKDVdhlyY-J5jgU8e6jHiDMviI/edit?usp=sharing)

---

## Overview

This project evaluates machine learning approaches for EEG-based stress classification. We implement four models — LDA, CNN, XGBoost, and Random Forest — across two EEG datasets, comparing classical linear methods against more complex ensemble and deep learning approaches.

Our results show that classical linear models (LDA) achieve modest performance with high inter-subject variability, while more complex models (XGBoost, CNN) demonstrate stronger and more consistent feature learning.

---

## Datasets

| Dataset | Used By | Task |
|---|---|---|
| DREAMER | LDA | Binary stress classification |
| SAM40 (`eeg_data.csv`) | CNN, XGBoost, Random Forest | 3-class stress classification |

---

## Models

### LDA (`LDA.ipynb`)
Binary stress classification on DREAMER using PSD-based bandpower features (theta, alpha, beta) across 14 EEG channels. Stress is defined as high arousal (≥3) and low valence (≤3). Three feature representations are compared: raw bandpower, baseline difference, and baseline log-ratio. Evaluated via subject-wise stratified 3-fold cross-validation.

### CNN (`EEG_Stress_Detection_CNN.ipynb`)
3-class stress classification on SAM40. A 783-feature vector per trial is reshaped into a 28×28 pseudo-image and passed through a 3-block convolutional network (32→64→128 filters) with LeakyReLU activations, trained for 100 epochs with Adam and categorical cross-entropy.

### XGBoost (`EEG_Stress_Detection_XGBoost.ipynb`)
3-class stress classification on SAM40. Gradient boosted trees (200 estimators, max depth 6, learning rate 0.1) with gain-based feature importances and SHAP explanations. Multi-class ROC and precision-recall curves computed via one-vs-rest binarization.

### Random Forest (`EEG_Stress_Detection_Random_Forest.ipynb`)
3-class stress classification on SAM40. Ensemble of 100 shallow trees (max depth 2) with feature importances.

---

## Results Summary

| Model | Task | Accuracy | Macro F1 |
|---|---|---|---|
| LDA | Binary stress (DREAMER) | ~0.58 | — |
| CNN | 3-class stress (SAM40) | 95.5% | 0.96 |
| XGBoost | 3-class stress (SAM40) | 98.4% | 0.98 |
| Random Forest | 3-class stress (SAM40) | 93.2% | 0.93 |

LDA performance was evaluated per-subject via cross-validation (mean AUC 0.54–0.58 across feature representations). CNN, XGBoost, and Random Forest were evaluated on a single 80/20 held-out test split.

---

## Setup

### Requirements
```
numpy pandas scipy scikit-learn matplotlib seaborn
tensorflow keras xgboost shap
```

### Data
- Place `DREAMER.mat` in the working directory before running `LDA.ipynb`
- Place `eeg_data.csv` in the working directory before running the CNN, XGBoost, and Random Forest notebooks
- The notebooks were originally developed in Google Colab — remove or update the `drive.mount` and `cd` cells if running locally

### Getting DREAMER
`DREAMER.mat` is not included in this repository due to file size. To obtain it:
1. Go to the [DREAMER dataset page on Zenodo](https://zenodo.org/record/546113)
2. Create a free Zenodo account if you don't already have one
3. Download `DREAMER.mat` and place it in the root of this repository before running `LDA.ipynb`

---

## References

1. Katmah et al. (2021). A Review on Mental Stress Assessment Methods Using EEG Signals. *Sensors*, 21(15), 5043.
2. Badr et al. (2024). A review on evaluating mental stress by deep learning using EEG signals. *Neural Computing and Applications*, 36, 12629–12654.
3. Katsigiannis & Ramzan (2018). DREAMER: A database for emotion recognition through EEG and ECG signals. *IEEE JBHI*, 22(1), 98–107.
4. Ghosh et al. (2022). SAM 40: Dataset of 40 subject EEG recordings to monitor induced-stress. *Data in Brief*, 40, 107772.
