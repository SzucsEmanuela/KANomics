import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import kan
from kan import KAN
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from kanomics import create_dataset_wo_outliers
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
import sympy
import os


def simple_linear_regression(dataset):
    print(f"Training Simple Linear Regression for {target_protein}...")

    # Initialize and train the Linear Regression model
    linear_model = LinearRegression()
    linear_model.fit(dataset['train_input'], dataset['train_label'])

    # Make predictions on the test set
    raw_preds_scaled_lr = linear_model.predict(dataset['test_input'])

    # Inverse transform to get original scale values for evaluation
    scaler_y = StandardScaler()
    y_scaled = torch.tensor(scaler_y.fit_transform(dataset['test_label']), dtype=torch.float32)
    y_test_unscaled_lr = scaler_y.inverse_transform(dataset['test_label'].numpy())
    raw_preds_unscaled_lr = scaler_y.inverse_transform(raw_preds_scaled_lr)

    # Flatten arrays for metrics calculation and plotting
    y_test_flat_lr = y_test_unscaled_lr.flatten()
    raw_preds_flat_lr = raw_preds_unscaled_lr.flatten()

    # Calculate Regression Metrics
    r2_lr = r2_score(y_test_flat_lr, raw_preds_flat_lr)
    mse_lr = mean_squared_error(y_test_flat_lr, raw_preds_flat_lr)
    mae_lr = mean_absolute_error(y_test_flat_lr, raw_preds_flat_lr)

    print(f"\n--- Simple Linear Regression Results ({target_protein}) ---")
    print(f"R2 Score: {r2_lr:.4f}")
    print(f"Mean Squared Error (MSE): {mse_lr:.4f}")
    print(f"Mean Absolute Error (MAE): {mae_lr:.4f}")

    # Plot Regression results
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#F8F9FA')
    ax.set_facecolor('#F8F9FA')
    sns.regplot(x=y_test_flat_lr, y=raw_preds_flat_lr, scatter_kws={'color': '#03807F', 'alpha': 0.3},
                line_kws={'color': '#F7B80E', 'linewidth': 5})
    plt.title(f'Linear regression, {target_protein} (R$^2$: {r2_lr:.2f}, MSE:{mse_lr:.4f})', fontsize=18)
    plt.xlabel(f'{target_protein} level', fontsize=16)
    plt.ylabel(f'Predicted {target_protein} level', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"./vib_linear_regression_{target_protein}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Linear regression plot saved")


def kan_regression(dataset, model):
    with torch.no_grad():
        raw_preds_scaled = model(dataset['test_input']).numpy()

        # Inverse transform to get original scale values for evaluation
        scaler_y = StandardScaler()
        y_scaled = torch.tensor(scaler_y.fit_transform(dataset['test_label']), dtype=torch.float32)
        y_test_unscaled = scaler_y.inverse_transform(dataset['test_label'].numpy())
        raw_preds_unscaled = scaler_y.inverse_transform(raw_preds_scaled)

    # Flatten arrays for metrics calculation and plotting
    y_test_flat = y_test_unscaled.flatten()
    raw_preds_flat = raw_preds_unscaled.flatten()

    # Calculate Regression Metrics
    r2 = r2_score(y_test_flat, raw_preds_flat)
    mse = mean_squared_error(y_test_flat, raw_preds_flat)
    mae = mean_absolute_error(y_test_flat, raw_preds_flat)
    print(f"R2 Score: {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    # 1. Set the outer figure background color
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#F8F9FA')

    # 2. Set the inner plot area background color
    ax.set_facecolor('#F8F9FA')

    # 3. Plot with custom colors for points and the regression line
    sns.regplot(
        x=y_test_flat,
        y=raw_preds_flat,
        scatter_kws={'color': '#03807F', 'alpha': 0.3},  # Teal points
        line_kws={'color': '#F7B80E', 'linewidth': 5},  # Yellow/Gold line
        ax=ax
    )

    plt.title(f'KAN regression, {target_protein} (R$^2$: {r2:.2f}, MSE:{mse:.4f})', fontsize=18)
    plt.xlabel(f'{target_protein} level', fontsize=16)
    plt.ylabel(f'Predicted {target_protein} level', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"./vib_kan_regression_plot_{target_protein}.png", dpi=300, bbox_inches='tight')
    plt.close()


def feature_importance_plot(in_feats, model):
    importances_v = model.feature_score.detach().cpu().numpy()
    norm_imp_v = 100 * importances_v / np.sum(importances_v)

    df_importance = pd.DataFrame({
        'Feature': in_feats,
        'Importance': norm_imp_v
    })
    #df_importance = df_importance.sort_values(by='Importance', ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#F8F9FA')
    ax.set_facecolor('#F8F9FA')

    plt.barh(df_importance['Feature'], df_importance['Importance'], color='#03807F')

    plt.title(f'{target_protein} feature attribution', fontsize=18, pad=15)
    plt.xlabel('Importance %', fontsize=14)
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)  # Grid lines on X-axis only makes it cleaner

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)
    plt.tight_layout()
    plt.savefig(f"./vib_after_prune_feature_importance_{target_protein}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_model(model):
    model.plot(folder="./vib_figures", scale=1.0)
    plt.savefig("kan_model_pruned_plot.png", dpi=300, bbox_inches='tight')
    plt.close()


def calculate_symbolic(model):
    #lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'abs']
    model.auto_symbolic()
    #model.fit(dataset, opt="Adam", steps=50, lr=0.1, lamb_entropy=10, loss_fn=nn.MSELoss())
    formula = model.symbolic_formula()[0][0]
    symb_exp = kan.utils.ex_round(formula, 4)
    print(f"Symbolic expression: {symb_exp}")
    latex_form = sympy.latex(symb_exp)
    print(f"LaTeX version: {latex_form}")
    with open('symbolic_expression_wo_lib.txt', mode='w') as f:
        f.write(str(latex_form))


if __name__ == '__main__':
    matplotlib.use('Agg')
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    torch.manual_seed(0)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    input_features = ['density', 'diversity', 'CD4', 'Cytokeratin', 'CD31', 'CD36', 'NKG2D', 'CDX2', 'ITLN1', 'CD68',
                      'CD34', 'CD117']
    target_protein = 'Ki67'
    df, dataset, in_features = create_dataset_wo_outliers("./23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv")
    model = KAN(width=[len(in_features), 5, 1], grid=20, k=3, seed=42)  # Note: grid should be the final grid used during training
    model.load_state_dict(torch.load("feats_5_1_grid20_kan_model_Ki67.pth"))
    model = model.to(device)
    model(dataset['train_input'])
    kan_regression(dataset, model)
    simple_linear_regression(dataset)
    #calculate_symbolic(model)
    # param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"The KAN model's parameter count: {param_count}")
