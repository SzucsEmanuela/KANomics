"""
Training pipeline for the baseline Multi-Layer Perceptron (MLP).
"""

from typing import Dict
import torch
import torch.nn as nn

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
    DATA_PATH = "./23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv"
    TARGET_PROTEIN = "Ki67"
    SAVE_PATH = "mlp_model_Ki67.pth"

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, dataset, in_features = load_kanomics_dataset(DATA_PATH)

    model = train_mlp_model(
        dataset=dataset,
        input_size=len(in_features),
        hidden_size=128,
        epochs=1000,
        lr=0.01,
        device=device,
    )

    # Save model weights for evaluation in gen_figures.py
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")
