# KANomics
Context-aware protein level prediction via Kolmogorov-Arnold Networks


[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![pykan](https://img.shields.io/badge/pykan-0.0.5+-brightgreen.svg)](https://github.com/KindXiaoming/pykan)

> **"KANs can learn interpretable, non-linear relationships between the cellular niche and the individual cell, revealing the math behind biology."**

## Overview & biological motivation
The molecular profile of a cell is not an isolated state, but a dynamic function of its spatial microenvironment. In complex tissues like the human colon, cellular behavior is governed by a sophisticated 'language' of neighborhood interactions. 

**KANomics** leverages **Kolmogorov-Arnold Networks (KANs)** to decode spatial proteomics data from high-plex **CODEX imaging**. By combining spatial feature engineering with interpretable spline-based neural activations, KANomics captures non-linear spatial dependencies while remaining explainable and transparent.

### Input data & preprocessing state
* **Source & Tissue:** Human colon tissue profiled via high-plex CODEX spatial proteomics.
* **Single-Cell Processing State:** The pipeline accepts pre-processed single-cell data (derived from cell segmentation). Input protein markers contain **pre-calculated, Z-normalized signal intensities** across individual cells alongside spatial $X,Y$ coordinates.
* **Spatial Feature Engineering:** KANomics builds upon these normalized intensities by querying spatial $k$-d trees (one per tissue region) to extract regional context features, such as local cell density, cell-type diversity, and neighborhood marker signal means. Coordinates are in pixels (~0.377 µm/px); the default radius of 100 px is about 38 µm. The centre cell is excluded from its own neighbourhood.

---

## Key highlights

* **Spatial microenvironment engineering:** Calculates local cell density, cell-type diversity, and neighborhood marker means using per-region spatial $k$-d trees.
* **Quality control:** Clips extreme antibody aggregates (quantile clipping, fitted on training cells) and prunes spatial density anomalies using a $3.0 \times \text{IQR}$ rule.
* **Predictive power:** In the preliminary (poster) setting, outperforms traditional Multi-Layer Perceptrons (MLPs) and Linear Models in predicting niche-level proliferation (e.g., **Ki67**); see the note below the benchmark.
* **Transparent architecture:** Retains full feature interpretability post-pruning and supports symbolic mathematical extraction via `pykan`.

---

## Performance benchmark

Preliminary results (conference poster) on predicting **niche-level Ki67**: the mean Ki67 level of the cells within ~38 µm of each cell, from neighbourhood means of other markers, using a random cell-level train/test split on $150,000$ colon cells:

| Model                            | $R^2$ score (Test) | MSE (Test)  |
|:---------------------------------|:------------------:|:-----------:|
| **Linear Regression**            |      $0.2900$      |  $0.7150$   |
| **Multi-Layer Perceptron (MLP)** |      $0.5800$      |  $0.4200$   |
| **KANomics (KAN)**               |     **0.7300**     | **0.2732**  |

> **Note:** these numbers come from the original pipeline, which predicted the neighbourhood mean rather than each cell's own Ki67, built neighbourhoods across overlapping donor regions, and evaluated on a random cell split. The pipeline now defaults to single-cell targets and leave-one-donor-out evaluation; see [`docs/pipeline_fixes.md`](docs/pipeline_fixes.md) for the changes and updated results.

---

## Feature attributions & biological context

The feature engineering pipeline generates 12 input features capturing microenvironment characteristics:

* **Spatial Features (Engineered):**
  * `nh_density`: Log-transformed local neighborhood cell density.
  * `nh_diversity`: Cell-type heterogeneity within the neighborhood radius.
* **Neighborhood Marker Signals:**
  * `CD4`, `Cytokeratin`, `CD31`, `CD36`, `NKG2D`, `CDX2`, `ITLN1`, `CD68`, `CD34`, `CD117`.

### Observation:
* **Top Attributions:** Feature attribution analysis reveals **CDX2** (intestinal differentiation) and **CD4** as the most critical predictors from this set of inputs for Ki67 proliferation dynamics.

---

## Repository structure

```text
KANomics/
├── kanomics/                  # Core Python library
│   ├── __init__.py
│   ├── config.py              # Repository paths and shared CLI options
│   ├── dataset.py             # Feature engineering, QC, train/test splits
│   ├── models.py              # Baseline model architectures (SimpleMLP)
│
├── scripts/                   # Execution scripts
│   ├── train_mlp.py           # Trains & evaluates baseline MLP
│   ├── train_kan.py           # KAN training, grid refinement, & pruning
│   └── gen_figures.py         # Visualizations & regression evaluation
│
├── docs/                      # Documentation & assets
│   ├── figures/               # Preliminary experiment plots
│   └── poster_VIB_Spatial_omics_2nd_edition_conference.pdf
│
├── requirements.txt           # Project dependencies
└── README.md

```

Run the scripts from the repository root, e.g. `PYTHONPATH=. python scripts/train_kan.py --target-mode cell --split donor --test-donor B004`. Models and figures are written to `outputs/`. Use `--target-mode niche --split cell` to reproduce the poster setting (with the per-region neighbourhood fix).
---

## Data availability

The spatial proteomics dataset used in this project originates from the **CODEX HuBMAP Dryad repository**:

1. Download the merged dataset (`23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv`, ~2.7 GB) from [Dryad Repository Link Here].
2. Place the `.csv` file into the root folder of this repository:
   ```bash
   KANomics/
   ├── 23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv
   ├── docs/
   ├── kanomics/
   ├── scripts/
   ├── requirements.txt
   └── README.md
   
---
## Next steps
KANomics is an evolving framework. Key ongoing and planned developments include:
* Hyperparameter tuning, finding an optimal and minimal architecture.
* Introducing an image analysis pipeline to utilize classical morphological features.

---
## References

1. Kolmogorov-Arnold Networks:

        Liu, Ziming (2025). KAN: Kolmogorov-Arnold Networks. 
        ICLR. https://doi.org/10.48550/arXiv.2404.19756

2. Spatial Proteomics Dataset:

        Hickey, John (2023). Processed single cell data from CODEX multiplexed imaging 
        of the human intestine [Dataset]. Dryad. https://doi.org/10.5061/dryad.pk0p2ngrf
