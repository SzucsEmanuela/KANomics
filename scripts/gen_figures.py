"""
Evaluation and figure generation module for KANomics.
Generates performance comparisons (KAN vs Linear Regression and MLP), feature
attribution plots, and symbolic regression outputs for protein targets.
"""

import os
from typing import Dict, List, Tuple
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import sympy
import torch

import kan
from kan import KAN
from kanomics.dataset import load_kanomics_dataset

matplotlib.use("Agg")

# Styling colour constants
PLOT_BG_COLOR = "#F8F9FA"
TEAL_COLOR = "#03807F"
GOLD_COLOR = "#F7B80E"


def evaluate_predictions(
        y_true: np.ndarray, y_pred: np.ndarray, model_name: str, target_protein: str) -> Tuple[float, float, float]:
    """Calculates regression metrics, prints them, and saves a standardized comparison plot.

    Args:
        y_true: Unscaled ground truth target values.
        y_pred: Unscaled model predictions.
        model_name: Name of the model (e.g., 'KAN', 'Linear Regression').
        target_protein: Name of the target protein being predicted.

    Returns:
        Tuple containing (R2 Score, MSE, MAE).
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    r2 = r2_score(y_true_flat, y_pred_flat)
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)

    print(f"\n--- {model_name} Results ({target_protein}) ---")
    print(f"R2 Score:                  {r2:.4f}")
    print(f"Mean Squared Error (MSE):  {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    sns.regplot(
        x=y_true_flat,
        y=y_pred_flat,
        scatter_kws={"color": TEAL_COLOR, "alpha": 0.3},
        line_kws={"color": GOLD_COLOR, "linewidth": 5},
        ax=ax,
    )

    plt.title(f"{model_name}, {target_protein} (R$^2$: {r2:.2f}, MSE: {mse:.4f})", fontsize=18)
    plt.xlabel(f"{target_protein} level", fontsize=16)
    plt.ylabel(f"Predicted {target_protein} level", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)

    output_filename = f"./vib_{model_name.lower().replace(' ', '_')}_{target_protein}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {output_filename}")

    return r2, mse, mae


def run_linear_regression(dataset: Dict[str, torch.Tensor], target_protein: str) -> None:
    """Trains a baseline Simple Linear Regression model and evaluates performance."""
    print(f"Training Simple Linear Regression for {target_protein}...")

    linear_model = LinearRegression()
    linear_model.fit(dataset["train_input"].numpy(), dataset["train_label"].numpy())

    raw_preds_scaled = linear_model.predict(dataset["test_input"].numpy())

    # fitting on training labels to prevent data leakage onto test evaluations
    scaler_y = StandardScaler()
    scaler_y.fit(dataset["train_label"].numpy())

    y_test_unscaled = scaler_y.inverse_transform(dataset["test_label"].numpy())
    raw_preds_unscaled = scaler_y.inverse_transform(raw_preds_scaled)

    evaluate_predictions(
        y_true=y_test_unscaled,
        y_pred=raw_preds_unscaled,
        model_name="Linear Regression",
        target_protein=target_protein,
    )


def run_mlp_regression(
    dataset: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    target_protein: str,
    device: str = "cpu",
) -> None:
    """Evaluates a trained MLP model on the test dataset."""
    model.eval()
    model.to(device)

    with torch.no_grad():
        test_inputs = dataset["test_input"].to(device)
        raw_preds_scaled = model(test_inputs).cpu().numpy()

    scaler_y = StandardScaler()
    scaler_y.fit(dataset["train_label"].numpy())

    y_test_unscaled = scaler_y.inverse_transform(dataset["test_label"].numpy())
    raw_preds_unscaled = scaler_y.inverse_transform(raw_preds_scaled)

    evaluate_predictions(
        y_true=y_test_unscaled,
        y_pred=raw_preds_unscaled,
        model_name="MLP Regression",
        target_protein=target_protein,
    )


def run_kan_regression(
    dataset: Dict[str, torch.Tensor], model: KAN, target_protein: str
) -> None:
    """Evaluates a pre-trained KAN model on the test dataset."""
    model.eval()
    with torch.no_grad():
        raw_preds_scaled = model(dataset["test_input"]).cpu().numpy()

    # fitting on training labels to prevent data leakage onto test evaluations
    scaler_y = StandardScaler()
    scaler_y.fit(dataset["train_label"].numpy())

    y_test_unscaled = scaler_y.inverse_transform(dataset["test_label"].numpy())
    raw_preds_unscaled = scaler_y.inverse_transform(raw_preds_scaled)

    evaluate_predictions(
        y_true=y_test_unscaled,
        y_pred=raw_preds_unscaled,
        model_name="KAN Regression",
        target_protein=target_protein,
    )


def plot_feature_importance(
    in_feats: List[str], model: KAN, target_protein: str, output_path: str = "."
) -> None:
    """Plots feature attributions extracted from the KAN feature score."""
    importances_v = model.feature_score.detach().cpu().numpy()
    norm_imp_v = 100 * importances_v / np.sum(importances_v)

    df_importance = pd.DataFrame({"Feature": in_feats, "Importance": norm_imp_v})

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=PLOT_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    plt.barh(df_importance["Feature"], df_importance["Importance"], color=TEAL_COLOR)

    plt.title(f"{target_protein} Feature Attribution", fontsize=18, pad=15)
    plt.xlabel("Importance %", fontsize=14)
    plt.grid(True, axis="x", linestyle="--", alpha=0.6)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", labelsize=14)
    plt.tight_layout()

    file_path = os.path.join(output_path, f"vib_feature_importance_{target_protein}.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Feature importance plot saved to: {file_path}")


def export_symbolic_expression(model: KAN, output_file: str = "symbolic_expression.txt") -> None:
    """Extracts mathematical symbolic expression from KAN and saves it as LaTeX."""
    model.auto_symbolic()
    formula = model.symbolic_formula()[0][0]
    symb_exp = kan.utils.ex_round(formula, 4)

    latex_form = sympy.latex(symb_exp)
    print(f"Extracted Symbolic Expression (LaTeX): {latex_form}")

    with open(output_file, mode="w") as f:
        f.write(str(latex_form))
    print(f"Symbolic formula saved to {output_file}")


if __name__ == "__main__":
    # Configuration
    DATA_PATH = "../23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv"
    MODEL_WEIGHTS_PATH = "../feats_5_1_grid20_kan_model_Ki67.pth"
    TARGET_PROTEIN = "Ki67"
    DEVICE = "cpu"

    torch.manual_seed(0)

    # Load data and initialized model
    df, dataset, in_features = load_kanomics_dataset(DATA_PATH)

    kan_model = KAN(width=[len(in_features), 5, 1], grid=20, k=3, seed=42)
    kan_model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
    kan_model = kan_model.to(DEVICE)

    # Dry run pass to initialize activation shapes
    kan_model(dataset["train_input"])

    # Run Evaluations & Plotting
    run_kan_regression(dataset, kan_model, TARGET_PROTEIN)
    run_linear_regression(dataset, TARGET_PROTEIN)
    plot_feature_importance(in_features, kan_model, TARGET_PROTEIN)
