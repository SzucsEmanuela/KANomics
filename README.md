# KANomics
Context-aware protein level prediction via Kolmogorov-Arnold Networks

The molecular profile of a cell does not only depend on itself, but it is a dynamic product of its spatial context. In complex tissues like the human colon, cellular behaviour is governed by a sophisticated 'language' of neighbourhood interactions. While traditional deep learning models have achieved high accuracy in decoding these language patterns, they often function as 'black boxes', obscuring the very biological rules they should uncover. Here, we introduce KANomics, a framework leveraging Kolmogorov-Arnold Networks (KANs) to predict proteome levels in a more transparent and explainable way. Using high-plex CODEX imaging data, we engineered niche-features such as local cell density, neighbourhood diversity and mean regional marker intensity to characterize the cellular microenvironment. We applied our method to the tasks of protein level prediction and cell type classification demonstrating the versatility of KANs across both continuous and categorical biological domains. In predicting functional markers such as Ki67, our method significantly outperformed simple linear regression, capturing non-linear spatial dependencies. While achieving predictive accuracy comparable to Multi-Layer Perceptrons (MLPs) of equivalent complexity, KANs offer a distinct advantage: the direct extraction of symbolic mathematical formulas that describe these biological relationships. By converting complex spatial patterns into interpretable equations, this framework moves us away from 'black-box' predictions toward a 'white-box' understanding of tissue structure and cellular function.

## Data Availability

The spatial proteomics dataset used in this project originates from the **CODEX HuBMAP Dryad repository**:

1. Download the merged dataset (`23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv`, ~2.7 GB) from [Dryad Repository Link Here].
2. Place the `.csv` file into the root folder of this repository:
   ```bash
   KANomics/
   ├── 23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv
   ├── kanomics/
   ├── scripts/
   └── README.md