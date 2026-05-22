# Coordinated-Client-Deformed-Prototype-Learning-for-Personalized-Federated-HAR

This repository contains the notebook implementation and benchmark analysis for **Coordinated Client-Deformed Prototype Learning (CDPL)** for personalized federated Human Activity Recognition (HAR).

CDPL is designed for subject-wise federated HAR, where each subject is treated as a client and raw sensor data remains local. The method learns a shared temporal encoder, a global class-prototype bank, and a low-rank deformation basis that allows each client or held-out subject to obtain personalized prototypes without fully retraining the global model.

The central personalization mechanism is:

```text
p_c^(i) = normalize(p_c + B a_c^(i))
```

where `p_c` is the global prototype for class `c`, `B` is the shared deformation basis, and `a_c^(i)` is the client-specific deformation coefficient. Classification is performed using cosine-style prototype logits:

```text
l_c^(i)(z) = tau_i * z^T p_c^(i)
```

where `z` is the normalized embedding and `tau_i` is the client-specific temperature.

---

## Repository Files

| File | Purpose |
|---|---|
| `CDPL Baseline Analysis.ipynb` | PAMAP2 baseline analysis notebook for running and evaluating the CDPL pipeline, including preprocessing, model training, personalization, and metric reporting. |
| `CDPL Benchmark Notebook.ipynb` | Combined benchmark notebook for comparing CDPL against federated learning baselines and producing benchmark outputs. |
| `CDPL PAMAP2 Statistical Benchmark Analysis.ipynb` | Statistical analysis notebook for PAMAP2 benchmark results, including per-subject/fold metrics, confidence intervals, and significance testing. |
| `CDPL PAMAP2 Training Notebook.ipynb` | Main PAMAP2 training notebook for the CDPL model using LOSO-style subject-level evaluation. |
| `CDPL UCI HAR Training Notebook.ipynb` | Training notebook adapted for the UCI HAR smartphone dataset. |

---

## Datasets

This repository uses two public HAR datasets from the UCI Machine Learning Repository.

### PAMAP2 Physical Activity Monitoring

- Dataset page: https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
- Expected default path in the PAMAP2 notebooks:

```text
/content/PAMAP2_Dataset/Protocol
```

The PAMAP2 notebooks use subject-wise client construction, activity filtering, sliding windows, and leakage-safe train/support/adaptation-validation/test partitioning.

### UCI HAR Using Smartphones

- Dataset page: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- Expected default path in the UCI HAR notebook:

```text
/content/drive/MyDrive/UCI HAR Dataset
```

The UCI HAR notebook uses the raw inertial signal windows from the dataset rather than only relying on the precomputed 561-feature vectors.

---

## Method Overview

CDPL follows a personalized federated learning setting for HAR:

1. Each subject is treated as a client.
2. A shared temporal encoder maps sensor windows into normalized embeddings.
3. A global prototype bank represents activity classes in the embedding space.
4. A shared low-rank basis models common directions of client heterogeneity.
5. Each client learns lightweight deformation coefficients to personalize the global prototypes.
6. For held-out subjects, only the personalization parameters are adapted using the support set while the global encoder remains fixed.

The model is intended to separate shared activity structure from subject-specific variation, which is important for HAR because sensor placement, body movement patterns, and subject physiology can vary across users.

---

## Main Components

### Temporal Encoder

The notebooks implement a temporal feature extractor with components such as:

- Temporal convolutional stem
- Positional encoding
- Transformer encoder layers
- Attentive temporal pooling
- Projection layer
- L2-normalized embedding output

### Personalized Prototype Classifier

The classifier uses client-deformed prototypes instead of a standard linear classification head. This allows personalization through a small number of deformation coefficients rather than full model fine-tuning.

### Federated Training

The training pipeline includes:

- Local client training
- Empirical prototype extraction
- Server-side aggregation
- Shared prototype and deformation-basis updates
- Held-out subject personalization
- Evaluation using Accuracy, Macro-F1, Expected Calibration Error, and Brier Score

---

## Suggested Notebook Order

For a full experiment workflow, run the notebooks in the following order:

1. `CDPL PAMAP2 Training Notebook.ipynb`  
   Train and evaluate the main CDPL model on PAMAP2.

2. `CDPL Baseline Analysis.ipynb`  
   Inspect baseline behavior and subject-wise performance trends.

3. `CDPL Benchmark Notebook.ipynb`  
   Run comparison experiments against benchmark methods.

4. `CDPL PAMAP2 Statistical Benchmark Analysis.ipynb`  
   Generate statistical results such as confidence intervals and paired significance tests.

5. `CDPL UCI HAR Training Notebook.ipynb`  
   Run the adapted CDPL pipeline on the UCI HAR smartphone dataset.

---

## Installation

The notebooks are designed to run in **Google Colab** or a Python environment with PyTorch support.

Install the required Python packages with:

```bash
pip install numpy pandas scikit-learn scipy matplotlib tqdm torch
```

For Google Colab, PyTorch and most scientific Python packages are usually preinstalled. If using Google Drive paths, mount Google Drive before running the dataset-loading cells.

---

## Running in Google Colab

1. Upload the notebooks to Google Colab.
2. Download the required dataset from the UCI repository.
3. Place the dataset in the path expected by the notebook, or update the `data_dir` field in the configuration cell.
4. Run the notebook cells sequentially.
5. Check the configured `save_dir` for generated CSV files, figures, and saved model artifacts.

Default output directories used in the notebooks include:

```text
./ccd_pamap2_runs
./cdpl_uci_har_runs
```

---

## Evaluation Metrics

The notebooks report the following metrics:

| Metric | Meaning |
|---|---|
| Accuracy | Overall fraction of correctly predicted activity windows. |
| Macro-F1 | Class-balanced F1 score, useful for activity imbalance. |
| ECE | Expected Calibration Error; lower values indicate better probability calibration. |
| Brier Score | Measures probabilistic prediction quality; lower values are better. |

For statistical reporting, use per-subject or per-fold values rather than only aggregate means.

---

## Notes on PAMAP2 LOSO Benchmarking

The PAMAP2 experiments are organized around leave-one-subject-out style evaluation. In benchmark and statistical comparisons, ensure that all methods use the same held-out subjects and the same preprocessing rules.

If a subject contains only one valid activity after filtering, exclude it from cross-method Macro-F1 and personalization comparisons, because single-class folds can make class-balanced comparisons misleading.

---

## Reproducibility

Most notebooks use a fixed random seed, commonly:

```text
seed = 42
```

For reproducible experiments:

- Keep dataset paths consistent.
- Use the same activity filtering rules.
- Use the same train/support/adaptation-validation/test splits.
- Run all baselines with the same held-out subjects.
- Save per-subject results before computing aggregate statistics.

---

## Recommended Repository Structure

```text
Coordinated-Client-Deformed-Prototype-Learning-for-Personalized-Federated-HAR/
│
├── README.md
├── CDPL Baseline Analysis.ipynb
├── CDPL Benchmark Notebook.ipynb
├── CDPL PAMAP2 Statistical Benchmark Analysis.ipynb
├── CDPL PAMAP2 Training Notebook.ipynb
├── CDPL UCI HAR Training Notebook.ipynb
│
├── results/                 # Optional: saved CSV files and tables
├── figures/                 # Optional: generated plots
└── saved_models/            # Optional: trained checkpoints
```

Dataset files are not included in this repository. Download them directly from the official UCI dataset pages.

---

## References

1. PAMAP2 Physical Activity Monitoring Dataset, UCI Machine Learning Repository:  
   https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

2. Human Activity Recognition Using Smartphones Dataset, UCI Machine Learning Repository:  
   https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

---

## Project Status

This repository is intended for research experimentation, benchmark reproduction, and manuscript-supporting analysis for personalized federated HAR using client-deformed prototype learning.
