"""
KAN (Kolmogorov-Arnold Network) training, grid refinement, and pruning pipeline.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

from kanomics.config import add_data_args, run_name
from kanomics import kan_compat  # noqa: F401  (CPU least squares for pykan on CUDA)
from kanomics.dataset import load_kanomics_dataset
from kan import KAN


def _get_kan_metrics(dataset: Dict[str, torch.Tensor], model: KAN) -> Tuple:
    """Helper to define evaluation metric closures required by pykan's fit loop."""

    def train_r2():
        with torch.no_grad():
            pred = model(dataset['train_input']).squeeze()
            target = dataset['train_label'].squeeze()
            return torch.tensor(r2_score(target.cpu().numpy(), pred.cpu().numpy()))

    def test_r2():
        with torch.no_grad():
            pred = model(dataset['test_input']).squeeze()
            target = dataset['test_label'].squeeze()
            return torch.tensor(r2_score(target.cpu().numpy(), pred.cpu().numpy()))

    def train_mae():
        with torch.no_grad():
            pred = model(dataset['train_input']).squeeze()
            target = dataset['train_label'].squeeze()
            return torch.mean(torch.abs(pred - target))

    def test_mae():
        with torch.no_grad():
            pred = model(dataset['test_input']).squeeze()
            target = dataset['test_label'].squeeze()
            return torch.mean(torch.abs(pred - target))

    def train_mse():
        with torch.no_grad():
            pred = model(dataset['train_input']).squeeze()
            target = dataset['train_label'].squeeze()
            return torch.mean((pred - target) ** 2)

    def test_mse():
        with torch.no_grad():
            pred = model(dataset['test_input']).squeeze()
            target = dataset['test_label'].squeeze()
            return torch.mean((pred - target) ** 2)

    return train_r2, train_mae, train_mse, test_r2, test_mae, test_mse


def train_kan_model(
    dataset: Dict[str, torch.Tensor],
    in_features: List[str],
    save_path: str = "kan_model_Ki67.pth",
) -> KAN:
    """Trains a KAN model with progressive grid refinement (grid 5 -> 10 -> 20)."""
    model = KAN(
        width=[len(in_features), 5, 1], grid=5, k=3, seed=42,
        device=dataset['train_input'].device, ckpt_path=str(Path(save_path).parent / 'pykan_ckpt'),
    )

    grid_schedule = [(5, 400), (10, 200), (20, 200)]

    for grid_size, steps in grid_schedule:
        if grid_size > 5:
            model = model.refine(grid_size)

        print(f"\n--- Training KAN with Grid Size: {grid_size} ({steps} steps) ---")
        metrics = _get_kan_metrics(dataset, model)

        results = model.fit(
            dataset,
            opt="Adam",
            steps=steps,
            lr=0.1,
            lamb_entropy=10,
            loss_fn=nn.MSELoss(),
            metrics=metrics,
            display_metrics=["train_r2", "test_r2"],
        )

        print(f"Train R2: {results['train_r2'][-1]:.4f} | Test R2: {results['test_r2'][-1]:.4f}")
        print(f"Train MSE: {results['train_mse'][-1]:.4f} | Test MSE: {results['test_mse'][-1]:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"KAN model saved to: {save_path}")
    return model


def fine_tune_kan_model(
    dataset: Dict[str, torch.Tensor],
    model_path: str,
    in_features: List[str],
    save_path: str = "kan_model_pruned.pth",
) -> KAN:
    """Loads a pre-trained KAN, prunes unused nodes, and refines grid resolution."""
    print(f"Loading pre-trained KAN from {model_path}...")
    device = dataset['train_input'].device
    model = KAN(
        width=[len(in_features), 5, 1], grid=20, k=3, seed=42,
        device=device, ckpt_path=str(Path(save_path).parent / 'pykan_ckpt'),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model(dataset['train_input'])  # Warmup activation shapes

    print("\n--- Pruning Model ---")
    model = model.prune()
    metrics = _get_kan_metrics(dataset, model)
    model.fit(
        dataset,
        opt="Adam",
        steps=100,
        lr=0.1,
        lamb_entropy=10,
        loss_fn=nn.MSELoss(),
        metrics=metrics,
    )

    print("\n--- Refining Pruned Model (Grid 40) ---")
    model = model.refine(40)
    metrics = _get_kan_metrics(dataset, model)
    results = model.fit(
        dataset,
        opt="Adam",
        steps=200,
        lr=0.1,
        lamb_entropy=10,
        loss_fn=nn.MSELoss(),
        metrics=metrics,
    )

    print(f"Final Pruned Test R2: {results['test_r2'][-1]:.4f}")
    torch.save(model.state_dict(), save_path)
    print(f"Pruned KAN saved to: {save_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_args(parser)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df, dataset, in_features, preprocessing = load_kanomics_dataset(
        args.data,
        target_protein=args.target,
        nrows=args.nrows,
        target_mode=args.target_mode,
        split=args.split,
        test_donor=args.test_donor,
        device=device,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = run_name(args.target, args.target_mode, args.split, preprocessing["test_donor"])
    kan_path = out_dir / f"kan_{name}.pth"

    # Train initial KAN
    kan_model = train_kan_model(dataset, in_features, save_path=kan_path)

    # Fine-tune / Prune
    fine_tune_kan_model(dataset, kan_path, in_features, save_path=out_dir / f"kan_{name}_pruned.pth")
