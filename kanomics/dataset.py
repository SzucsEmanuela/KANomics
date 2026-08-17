"""
Data processing pipeline for spatial proteomics (CODEX/HuBMAP data).
Handles KDTree spatial feature engineering (neighborhood density, cell type diversity,
neighbor marker averages), outlier clipping, and PyTorch dataset splits.

Public dataset source: https://datadryad.org/dataset/doi:10.5061/dryad.pk0p2ngrf
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch

DEFAULT_CONTEXT_MARKERS = [
    'CD4', 'Cytokeratin', 'CD31', 'CD36', 'NKG2D', 'CDX2', 'ITLN1', 'CD68', 'CD34', 'CD117'
]


def load_kanomics_dataset(
    csv_path: str,
    target_protein: str = 'Ki67',
    context_markers: List[str] = None,
    nrows: int = 150000,
    radius: float = 100.0,
    test_size: float = 0.3,
    random_state: int = 42,
    device: str = 'cpu',
) -> Tuple[pd.DataFrame, Dict[str, torch.Tensor], List[str]]:
    """Loads spatial proteomics CSV data, builds spatial features via KDTree, cleans

    outliers, scales features, and splits into train/test PyTorch tensors.

    Args:
        csv_path: Path to raw CODEX/HuBMAP CSV data.
        target_protein: Target marker to predict (e.g. 'Ki67').
        context_markers: List of protein markers for neighborhood averaging.
        nrows: Maximum rows to read from CSV.
        radius: Spatial neighborhood search radius (microenvironment distance).
        test_size: Fraction of data reserved for testing.
        random_state: Random seed for reproducibility.
        device: PyTorch device string ('cpu' or 'cuda').

    Returns:
        Tuple of (cleaned_dataframe, dataset_dict, feature_names_list).
    """
    if context_markers is None:
        context_markers = DEFAULT_CONTEXT_MARKERS

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, nrows=nrows)
    df_cleaned = df.copy()

    # Build spatial KDTree based on X, Y coordinates
    coords = df_cleaned[['Xcorr', 'Ycorr']].values
    tree = KDTree(coords)
    indices = tree.query_ball_point(coords, r=radius)

    print("Engineering spatial microenvironment features...")

    # 1. Biological context: Clip antibody outlier aggregates & compute neighbor means
    for marker in context_markers + [target_protein]:
        marker_values = df[marker].values
        df_cleaned[f'neigh_mean_{marker}'] = [
            np.mean(marker_values[idx]) if len(idx) > 0 else 0 for idx in indices
        ]
        upper_limit = df_cleaned[f'neigh_mean_{marker}'].quantile(0.999)
        lower_limit = df_cleaned[f'neigh_mean_{marker}'].quantile(0.001)
        df_cleaned[marker] = np.clip(
            df_cleaned[f'neigh_mean_{marker}'], lower_limit, upper_limit
        )

    # 2. Local cell density feature
    df_cleaned['nh_density'] = tree.query_ball_point(coords, r=radius, return_length=True)
    df_cleaned['nh_density'] = np.log1p(df_cleaned['nh_density'])

    # Filter extreme structural density outliers (3.0 * IQR)
    q1 = df_cleaned['nh_density'].quantile(0.25)
    q3 = df_cleaned['nh_density'].quantile(0.75)
    iqr = q3 - q1
    df_cleaned = df_cleaned[df_cleaned['nh_density'] <= (q3 + 3.0 * iqr)]

    # 3. Cell type diversity in spatial neighborhood
    cell_types = df['Cell Type'].values
    df_cleaned['nh_diversity'] = [
        len(np.unique(cell_types[idx])) for idx in tree.query_ball_point(coords, r=radius)
    ]

    feature_names = ['nh_density', 'nh_diversity'] + context_markers
    df_cleaned = df_cleaned.dropna(subset=feature_names + [target_protein])

    # Extract feature matrices and target vectors
    X = df_cleaned[feature_names].values
    y = df_cleaned[target_protein].values.reshape(-1, 1)

    # Scale inputs & targets
    scaler_X = MinMaxScaler()
    X_scaled = torch.tensor(scaler_X.fit_transform(X), dtype=torch.float32)

    scaler_y = StandardScaler()
    y_scaled = torch.tensor(scaler_y.fit_transform(y), dtype=torch.float32)

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=random_state, shuffle=True
    )

    dataset = {
        'train_input': X_train.to(device),
        'train_label': y_train.to(device),
        'test_input': X_test.to(device),
        'test_label': y_test.to(device),
    }

    print(f"Dataset ready. Train size: {len(X_train)}, Test size: {len(X_test)}")
    return df_cleaned, dataset, feature_names