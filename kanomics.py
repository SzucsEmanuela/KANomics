import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from kan import KAN
from kan.utils import ex_round
from scipy.spatial import KDTree
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score


def create_dataset(csv_path):
    df = pd.read_csv(csv_path, nrows=150000)
    target_protein = 'Ki67'
    context_markers = ['CD4', 'Cytokeratin', 'CD31', 'CD36', 'NKG2D', 'CDX2', 'ITLN1', 'CD68', 'CD34', 'CD117']
    df[target_protein].dropna().to_numpy()

    print(f"Engineering spatial features for {target_protein} prediction...")
    coords = df[['Xcorr', 'Ycorr']].values
    tree = KDTree(coords)

    # Geometric context
    df['local_density'] = tree.query_ball_point(coords, r=100, return_length=True)
    cell_types = df['Cell Type'].values
    df['type_diversity'] = [len(np.unique(cell_types[idx])) for idx in tree.query_ball_point(coords, r=100)]

    # Biological context (neighbor signaling)
    print(f"Engineering biological context features for {target_protein} prediction...")
    indices = tree.query_ball_point(coords, r=100)
    for marker in context_markers:
        marker_values = df[marker].values
        df[f'neigh_mean_{marker}'] = [
            np.mean(marker_values[idx]) if len(idx) > 0 else 0
            for idx in indices
        ]

    features = ['local_density', 'type_diversity'] + [f'neigh_mean_{m}' for m in context_markers]
    df = df.dropna(subset=features + [target_protein])

    X = df[features].values
    y = df[target_protein].values.reshape(-1, 1)

    scaler_X = MinMaxScaler()
    X_scaled = torch.tensor(scaler_X.fit_transform(X), dtype=torch.float32)

    scaler_y = StandardScaler()
    y_scaled = torch.tensor(scaler_y.fit_transform(y), dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42,
                                                        shuffle=True)

    dataset = {'train_input': X_train, 'train_label': y_train,
               'test_input': X_test, 'test_label': y_test}

    return df, dataset, features


def create_dataset_wo_outliers(csv_path):
    df = pd.read_csv(csv_path, nrows=150000)
    target_protein = 'Ki67'
    context_markers = ['CD4', 'Cytokeratin', 'CD31', 'CD36', 'NKG2D', 'CDX2', 'ITLN1', 'CD68', 'CD34', 'CD117']
    df[target_protein].dropna().to_numpy()

    df_cleaned = df.copy()
    coords = df_cleaned[['Xcorr', 'Ycorr']].values
    tree = KDTree(coords)

    # Clip extreme Z-scores to the 99.9th percentile to remove antibody aggregates
    indices = tree.query_ball_point(coords, r=100)
    for marker in context_markers + [target_protein]:
        marker_values = df[marker].values
        df_cleaned[f'neigh_mean_{marker}'] = [np.mean(marker_values[idx]) if len(idx) > 0 else 0 for idx in indices]
        upper_limit = df_cleaned[f'neigh_mean_{marker}'].quantile(0.999)
        lower_limit = df_cleaned[f'neigh_mean_{marker}'].quantile(0.001)
        df_cleaned[marker] = np.clip(df_cleaned[f'neigh_mean_{marker}'], lower_limit, upper_limit)

    df_cleaned['nh_density'] = tree.query_ball_point(coords, r=100, return_length=True)
    # Log-transforming dampens extreme density variations before MinMaxScaler.
    df_cleaned['nh_density'] = np.log1p(df_cleaned['nh_density'])

    # Drop extreme structural outliers using the IQR rule on the transformed density
    q1 = df_cleaned['nh_density'].quantile(0.25)
    q3 = df_cleaned['nh_density'].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + (3.0 * iqr)  # 3.0 IQR strictly targets extreme anomalies

    df_cleaned = df_cleaned[df_cleaned['nh_density'] <= upper_bound]

    cell_types = df['Cell Type'].values
    df_cleaned['nh_diversity'] = [len(np.unique(cell_types[idx])) for idx in tree.query_ball_point(coords, r=100)]

    all_inputs = ['nh_density', 'nh_diversity'] + context_markers
    df_cleaned = df_cleaned.dropna(subset=all_inputs)

    X = df_cleaned[all_inputs].values
    y = df_cleaned[target_protein].values.reshape(-1, 1)

    scaler_X = MinMaxScaler()
    X_scaled = torch.tensor(scaler_X.fit_transform(X), dtype=torch.float32)
    scaler_y = StandardScaler()
    y_scaled = torch.tensor(scaler_y.fit_transform(y), dtype=torch.float32)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42, shuffle=True)
    dataset = {'train_input': X_train.to(device), 'train_label': y_train.to(device),
               'test_input': X_test.to(device), 'test_label': y_test.to(device)}

    return df, dataset, all_inputs


def train_kan(dataset, in_feats):
    def train_r2():
        with torch.no_grad():
            pred = model(dataset["train_input"]).squeeze()
            target = dataset["train_label"].squeeze()
            return torch.tensor(r2_score(target.cpu().numpy(), pred.cpu().numpy()))

    def test_r2():
        with torch.no_grad():
            pred = model(dataset["test_input"]).squeeze()
            target = dataset["test_label"].squeeze()
            return torch.tensor(r2_score(target.cpu().numpy(), pred.cpu().numpy()))

    def train_mae():
        with torch.no_grad():
            pred = model(dataset["train_input"]).squeeze()
            target = dataset["train_label"].squeeze()
            return torch.mean(torch.abs(pred - target))

    def test_mae():
        with torch.no_grad():
            pred = model(dataset["test_input"]).squeeze()
            target = dataset["test_label"].squeeze()
            return torch.mean(torch.abs(pred - target))

    def train_mse():
        with torch.no_grad():
            pred = model(dataset["train_input"]).squeeze()
            target = dataset["train_label"].squeeze()
            return torch.mean((pred - target) ** 2)

    def test_mse():
        with torch.no_grad():
            pred = model(dataset["test_input"]).squeeze()
            target = dataset["test_label"].squeeze()
            return torch.mean((pred - target) ** 2)

    dtype = torch.get_default_dtype()
    model = KAN(width=[len(in_feats), 5, 1], grid=5, k=3, seed=42)
    results = model.fit(dataset, opt="Adam", steps=400, lr=0.1, lamb_entropy=10, loss_fn=nn.MSELoss(),
                        metrics=(train_r2, train_mae, train_mse, test_r2, test_mae, test_mse),
                        display_metrics=["train_r2", "test_r2"])
    print(f"train r2: {results['train_r2'][-1]:.4f}, test r2: {results['test_r2'][-1]:.4f}")
    print(f"train mae: {results['train_mae'][-1]:.4f}, test mae: {results['test_mae'][-1]:.4f}")
    print(f"train mse: {results['train_mse'][-1]:.4f}, test mse: {results['test_mse'][-1]:.4f}")
    print("----------------------------------------------------------------------------")

    model = model.refine(10)
    results = model.fit(dataset, opt="Adam", steps=200, lr=0.1, lamb_entropy=10, loss_fn=nn.MSELoss(),
                        metrics=(train_r2, train_mae, train_mse, test_r2, test_mae, test_mse),
                        display_metrics=["train_r2", "test_r2"])
    print(f"train r2: {results['train_r2'][-1]:.4f}, test r2: {results['test_r2'][-1]:.4f}")
    print(f"train mae: {results['train_mae'][-1]:.4f}, test mae: {results['test_mae'][-1]:.4f}")
    print(f"train mse: {results['train_mse'][-1]:.4f}, test mse: {results['test_mse'][-1]:.4f}")
    print("----------------------------------------------------------------------------")

    model = model.refine(20)
    results = model.fit(dataset, opt="Adam", steps=200, lr=0.1, lamb_entropy=10, loss_fn=nn.MSELoss(),
                        metrics=(train_r2, train_mae, train_mse, test_r2, test_mae, test_mse),
                        display_metrics=["train_r2", "test_r2"])
    print(f"train r2: {results['train_r2'][-1]:.4f}, test r2: {results['test_r2'][-1]:.4f}")
    print(f"train mae: {results['train_mae'][-1]:.4f}, test mae: {results['test_mae'][-1]:.4f}")
    print(f"train mse: {results['train_mse'][-1]:.4f}, test mse: {results['test_mse'][-1]:.4f}")

    model_save_path = f"./feats_5_1_grid20_kan_model_Ki67.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"KAN model saved to: {model_save_path}")


def fine_tune_kan_model(dataset, model_path, in_feats):
    def train_r2():
        with torch.no_grad():
            pred = model(dataset["train_input"]).squeeze()
            target = dataset["train_label"].squeeze()
            return torch.tensor(r2_score(target.cpu().numpy(), pred.cpu().numpy()))

    def test_r2():
        with torch.no_grad():
            pred = model(dataset["test_input"]).squeeze()
            target = dataset["test_label"].squeeze()
            return torch.tensor(r2_score(target.cpu().numpy(), pred.cpu().numpy()))

    def train_mae():
        with torch.no_grad():
            pred = model(dataset["train_input"]).squeeze()
            target = dataset["train_label"].squeeze()
            return torch.mean(torch.abs(pred - target))

    def test_mae():
        with torch.no_grad():
            pred = model(dataset["test_input"]).squeeze()
            target = dataset["test_label"].squeeze()
            return torch.mean(torch.abs(pred - target))

    def train_mse():
        with torch.no_grad():
            pred = model(dataset["train_input"]).squeeze()
            target = dataset["train_label"].squeeze()
            return torch.mean((pred - target) ** 2)

    def test_mse():
        with torch.no_grad():
            pred = model(dataset["test_input"]).squeeze()
            target = dataset["test_label"].squeeze()
            return torch.mean((pred - target) ** 2)

    model = KAN(width=[len(in_feats), 5, 1], grid=20, k=3, seed=42) # Note: grid should be the final grid used during training
    model.load_state_dict(torch.load(model_path))
    model(dataset['train_input'])

    model = model.prune()
    results = model.fit(dataset, opt="Adam", steps=100, lr=0.1, lamb_entropy=10, loss_fn=nn.MSELoss(),
                        metrics=(train_r2, train_mae, train_mse, test_r2, test_mae, test_mse),
                        display_metrics=["train_r2", "test_r2"])
    print(f"train r2: {results['train_r2'][-1]:.4f}, test r2: {results['test_r2'][-1]:.4f}")
    print(f"train mae: {results['train_mae'][-1]:.4f}, test mae: {results['test_mae'][-1]:.4f}")
    print(f"train mse: {results['train_mse'][-1]:.4f}, test mse: {results['test_mse'][-1]:.4f}")

    model = model.refine(40)
    results = model.fit(dataset, opt="Adam", steps=200, lr=0.1, lamb_entropy=10, loss_fn=nn.MSELoss(),
                        metrics=(train_r2, train_mae, train_mse, test_r2, test_mae, test_mse),
                        display_metrics=["train_r2", "test_r2"])
    print(f"train r2: {results['train_r2'][-1]:.4f}, test r2: {results['test_r2'][-1]:.4f}")
    print(f"train mae: {results['train_mae'][-1]:.4f}, test mae: {results['test_mae'][-1]:.4f}")
    print(f"train mse: {results['train_mse'][-1]:.4f}, test mse: {results['test_mse'][-1]:.4f}")

    model_save_path = f"./{len(in_feats)}_5_1_grid40_Ki67_pruned.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"KAN model saved to: {model_save_path}")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    df, dataset, in_feats = create_dataset_wo_outliers("./23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv")
    fine_tune_kan_model(dataset, "./feats_5_1_grid20_kan_model_Ki67.pth", in_feats)
    #train_kan(dataset, in_feats)
