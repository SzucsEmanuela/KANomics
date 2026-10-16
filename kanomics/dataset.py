"""
Data processing pipeline for spatial proteomics (CODEX/HuBMAP data).
Handles KDTree spatial feature engineering (neighborhood density, cell type diversity,
neighbor marker averages), outlier clipping, and PyTorch dataset splits.

Spatial units: `Xcorr`/`Ycorr` are in pixels (~0.377 um/px), so radius=100 is ~38 um.
Regions of different donors share one tiled coordinate canvas and overlap, therefore
neighbourhoods are always computed within a single `unique_region`.

Public dataset source: https://datadryad.org/dataset/doi:10.5061/dryad.pk0p2ngrf
"""

from typing import Dict, Iterator, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import KDTree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch

DEFAULT_CONTEXT_MARKERS = [
    'CD4', 'Cytokeratin', 'CD31', 'CD36', 'NKG2D', 'CDX2', 'ITLN1', 'CD68', 'CD34', 'CD117'
]
SPATIAL_FEATURES = ['nh_density', 'nh_diversity']
FEATURE_SETS = ('spatial', 'niche', 'own')
TARGET_MODES = ('cell', 'niche')
SPLITS = ('donor', 'cell')

CLIP_QUANTILES = (0.001, 0.999)


def _neighbour_graph(df: pd.DataFrame, radius: float) -> sparse.csr_matrix:
    """Builds a binary cell-cell adjacency matrix (self excluded), one KDTree per region."""
    rows, cols = [], []
    positions = np.arange(len(df))
    for _, region_pos in pd.Series(positions).groupby(df['unique_region'].values):
        region_pos = region_pos.values
        coords = df[['Xcorr', 'Ycorr']].values[region_pos]
        pairs = KDTree(coords).query_pairs(r=radius, output_type='ndarray')
        rows.append(region_pos[pairs[:, 0]])
        cols.append(region_pos[pairs[:, 1]])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    adj = sparse.coo_matrix(
        (np.ones(2 * len(rows), dtype=np.float32), (np.r_[rows, cols], np.r_[cols, rows])),
        shape=(len(df), len(df)),
    )
    return adj.tocsr()


def build_features(
    csv_path: str,
    target_protein: str = 'Ki67',
    context_markers: Optional[List[str]] = None,
    nrows: Optional[int] = 150000,
    radius: float = 100.0,
    target_mode: str = 'cell',
    feature_sets: Sequence[str] = ('spatial', 'niche'),
    density_iqr_factor: Optional[float] = 3.0,
) -> Tuple[pd.DataFrame, List[str], str]:
    """Loads the CSV and engineers per-cell spatial microenvironment features.

    All neighbourhood features exclude the centre cell and only use cells of the same
    `unique_region`. Source marker columns are never overwritten.

    Args:
        csv_path: Path to raw CODEX/HuBMAP CSV data.
        target_protein: Target marker to predict (e.g. 'Ki67').
        context_markers: Markers used as inputs. The target is always removed from this list.
        nrows: Maximum rows to read from CSV (None reads everything).
        radius: Neighbourhood search radius in pixels (~0.377 um/px).
        target_mode: 'cell' predicts the cell's own target level, 'niche' predicts the mean
            target level of its neighbours (the preliminary poster setting).
        feature_sets: Any of 'spatial' (nh_density, nh_diversity), 'niche' (neighbour means
            of the context markers) and 'own' (the cell's own context marker levels).
        density_iqr_factor: Drops cells with nh_density above Q3 + factor * IQR, after all
            features are computed. None disables the filter.

    Returns:
        Tuple of (feature_dataframe, feature_names_list, target_column_name).
    """
    if target_mode not in TARGET_MODES:
        raise ValueError(f"target_mode must be one of {TARGET_MODES}, got {target_mode!r}")
    unknown_sets = set(feature_sets) - set(FEATURE_SETS)
    if unknown_sets:
        raise ValueError(f"Unknown feature_sets {unknown_sets}; choose from {FEATURE_SETS}")
    if context_markers is None:
        context_markers = DEFAULT_CONTEXT_MARKERS
    context_markers = [m for m in context_markers if m != target_protein]

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, nrows=nrows)
    df = df.dropna(subset=['Xcorr', 'Ycorr', 'unique_region', 'donor', target_protein])
    df = df.reset_index(drop=True)

    print("Engineering spatial microenvironment features (per region, centre cell excluded)...")
    adj = _neighbour_graph(df, radius)
    n_neigh = np.asarray(adj.sum(axis=1)).ravel()

    # 1. Neighbourhood marker means (NaN for cells without neighbours)
    with np.errstate(invalid='ignore', divide='ignore'):
        for marker in context_markers + [target_protein]:
            df[f'neigh_mean_{marker}'] = (adj @ df[marker].values) / n_neigh

    # 2. Local cell density feature
    df['nh_density'] = np.log1p(n_neigh)

    # 3. Cell type diversity: number of distinct cell types among the neighbours
    type_codes, _ = pd.factorize(df['Cell Type'])
    type_onehot = sparse.csr_matrix(
        (np.ones(len(df), dtype=np.float32), (np.arange(len(df)), type_codes))
    )
    df['nh_diversity'] = np.asarray(((adj @ type_onehot) > 0).sum(axis=1)).ravel()

    # Row filters, applied only after all features are computed
    isolated = n_neigh == 0
    if isolated.any():
        print(f"Dropping {isolated.sum()} cells without neighbours within r={radius}.")
        df = df[~isolated]

    if density_iqr_factor is not None:
        q1, q3 = df['nh_density'].quantile([0.25, 0.75])
        dense = df['nh_density'] > q3 + density_iqr_factor * (q3 - q1)
        print(f"Density IQR filter ({density_iqr_factor} x IQR) removed {dense.sum()} cells.")
        df = df[~dense]

    feature_names = []
    if 'spatial' in feature_sets:
        feature_names += SPATIAL_FEATURES
    if 'niche' in feature_sets:
        feature_names += [f'neigh_mean_{m}' for m in context_markers]
    if 'own' in feature_sets:
        feature_names += context_markers

    target_col = target_protein if target_mode == 'cell' else f'neigh_mean_{target_protein}'
    df = df.dropna(subset=feature_names + [target_col]).reset_index(drop=True)

    print(f"{len(df)} cells, {df['donor'].nunique()} donors, {len(feature_names)} features, "
          f"target '{target_col}'.")
    return df, feature_names, target_col


def make_split(
    df: pd.DataFrame,
    feature_names: List[str],
    target_col: str,
    split: str = 'donor',
    test_donor: Optional[str] = None,
    test_size: float = 0.3,
    random_state: int = 42,
    device: str = 'cpu',
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    """Splits cells into train/test, then clips and scales using training cells only.

    Args:
        df: Output of `build_features`.
        feature_names: Input columns.
        target_col: Target column.
        split: 'donor' holds out all cells of `test_donor` (leave-one-donor-out);
            'cell' is a random cell-level split (spatially autocorrelated, optimistic).
        test_donor: Donor held out when split='donor'. Defaults to the first donor
            in sorted order.
        test_size: Test fraction for split='cell'.
        random_state: Random seed for split='cell'.
        device: PyTorch device string ('cpu' or 'cuda').

    Returns:
        Tuple of (pykan dataset dict, fitted preprocessing dict with keys
        'X', 'y' (scalers), 'clip_X', 'clip_y' (quantile bounds), 'test_mask',
        'test_donor' (None for split='cell')).
    """
    if split == 'donor':
        donors = sorted(df['donor'].unique())
        if test_donor is None:
            test_donor = donors[0]
        if test_donor not in donors:
            raise ValueError(f"test_donor {test_donor!r} not in {donors}")
        test_mask = (df['donor'] == test_donor).values
    elif split == 'cell':
        _, test_idx = train_test_split(
            np.arange(len(df)), test_size=test_size, random_state=random_state, shuffle=True
        )
        test_mask = np.zeros(len(df), dtype=bool)
        test_mask[test_idx] = True
    else:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    X = df[feature_names].values.astype(np.float64)
    y = df[[target_col]].values.astype(np.float64)
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]

    # Clip antibody outlier aggregates with training quantiles
    clip_X = np.quantile(X_train, CLIP_QUANTILES, axis=0)
    clip_y = np.quantile(y_train, CLIP_QUANTILES, axis=0)
    X_train, X_test = (np.clip(a, clip_X[0], clip_X[1]) for a in (X_train, X_test))
    y_train, y_test = (np.clip(a, clip_y[0], clip_y[1]) for a in (y_train, y_test))

    scaler_X = MinMaxScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    def to_tensor(a: np.ndarray) -> torch.Tensor:
        return torch.tensor(a, dtype=torch.float32, device=device)

    dataset = {
        'train_input': to_tensor(scaler_X.transform(X_train)),
        'train_label': to_tensor(scaler_y.transform(y_train)),
        'test_input': to_tensor(scaler_X.transform(X_test)),
        'test_label': to_tensor(scaler_y.transform(y_test)),
    }
    preprocessing = {
        'X': scaler_X, 'y': scaler_y, 'clip_X': clip_X, 'clip_y': clip_y, 'test_mask': test_mask,
        'test_donor': test_donor if split == 'donor' else None,
    }

    held_out = f"donor {test_donor}" if split == 'donor' else "random cells"
    print(f"Dataset ready ({split} split, test = {held_out}). "
          f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return dataset, preprocessing


def iter_donor_folds(
    df: pd.DataFrame, feature_names: List[str], target_col: str, device: str = 'cpu'
) -> Iterator[Tuple[str, Dict[str, torch.Tensor], Dict[str, object]]]:
    """Yields (test_donor, dataset, preprocessing) for leave-one-donor-out CV."""
    for donor in sorted(df['donor'].unique()):
        dataset, preprocessing = make_split(
            df, feature_names, target_col, split='donor', test_donor=donor, device=device
        )
        yield donor, dataset, preprocessing


def load_kanomics_dataset(
    csv_path: str,
    target_protein: str = 'Ki67',
    context_markers: Optional[List[str]] = None,
    nrows: Optional[int] = 150000,
    radius: float = 100.0,
    target_mode: str = 'cell',
    feature_sets: Sequence[str] = ('spatial', 'niche'),
    split: str = 'donor',
    test_donor: Optional[str] = None,
    test_size: float = 0.3,
    random_state: int = 42,
    device: str = 'cpu',
) -> Tuple[pd.DataFrame, Dict[str, torch.Tensor], List[str], Dict[str, object]]:
    """Convenience wrapper: `build_features` followed by a single `make_split`.

    Returns:
        Tuple of (feature_dataframe, dataset_dict, feature_names_list, preprocessing_dict).
    """
    df, feature_names, target_col = build_features(
        csv_path,
        target_protein=target_protein,
        context_markers=context_markers,
        nrows=nrows,
        radius=radius,
        target_mode=target_mode,
        feature_sets=feature_sets,
    )
    dataset, preprocessing = make_split(
        df,
        feature_names,
        target_col,
        split=split,
        test_donor=test_donor,
        test_size=test_size,
        random_state=random_state,
        device=device,
    )
    return df, dataset, feature_names, preprocessing
