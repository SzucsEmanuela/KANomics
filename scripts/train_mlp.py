"""
Training pipeline for the baseline Multi-Layer Perceptron (MLP).
"""

import argparse
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn

from kanomics.config import add_data_args, run_name
from kanomics.dataset import load_kanomics_dataset
from kanomics.models import SimpleMLP


def train_mlp_model(
    dataset: Dict[str, torch.Tensor],
    input_size: int,
    hidden_size: int = 128,
    epochs: int = 1000,
    lr: float = 0.01,
    device: str = "cpu",
) -> SimpleMLP:
    """Trains a SimpleMLP model on the provided dataset."""
    mlp_model = SimpleMLP(input_size, hidden_size, output_size=1).to(device)
    print(f"MLP Parameter Count: {mlp_model.count_parameters()}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=lr)

    train_inputs = dataset["train_input"].to(device)
    train_labels = dataset["train_label"].to(device)

    print("Training Simple MLP...")
    for epoch in range(epochs):
        mlp_model.train()
        optimizer.zero_grad()

        outputs = mlp_model(train_inputs)
        loss = criterion(outputs, train_labels)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    print("MLP training complete.")
    return mlp_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_args(parser)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, dataset, in_features, preprocessing = load_kanomics_dataset(
        args.data,
        target_protein=args.target,
        nrows=args.nrows,
        target_mode=args.target_mode,
        split=args.split,
        test_donor=args.test_donor,
    )

    model = train_mlp_model(
        dataset=dataset,
        input_size=len(in_features),
        hidden_size=128,
        epochs=1000,
        lr=0.01,
        device=device,
    )

    # Save model weights for evaluation in gen_figures.py
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"mlp_{run_name(args.target, args.target_mode, args.split, preprocessing['test_donor'])}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
