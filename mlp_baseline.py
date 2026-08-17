import torch.nn as nn
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from kanomics import create_dataset_wo_outliers
import os


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def mlp(device, dataset):
    target_protein = 'Ki67'
    mlp_model = SimpleMLP(12, 128, 1)
    mlp_model.to(device)
    param_count = sum(p.numel() for p in mlp_model.parameters() if p.requires_grad)
    print(f"The MLP model's parameter count: {param_count}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.01)

    print(f"Training Simple MLP for Ki67...")

    num_epochs = 1000

    train_losses_mlp = []

    for epoch in range(num_epochs):
        mlp_model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = mlp_model(dataset['train_input'].to(device))
        loss = criterion(outputs, dataset['train_label'].to(device))

        # Backward and optimize
        loss.backward()
        optimizer.step()

        train_losses_mlp.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("MLP training complete.")

    mlp_model.eval()
    with torch.no_grad():
        raw_preds_scaled_mlp = mlp_model(dataset['test_input'].to(device)).cpu().numpy()

        # Inverse transform to get original scale values for evaluation
        scaler_y = StandardScaler()
        y_scaled = torch.tensor(scaler_y.fit_transform(dataset['test_label']), dtype=torch.float32)
        y_test_unscaled_mlp = scaler_y.inverse_transform(dataset['test_label'].cpu().numpy())
        raw_preds_unscaled_mlp = scaler_y.inverse_transform(raw_preds_scaled_mlp)

    # Flatten arrays for metrics calculation and plotting
    y_test_flat_mlp = y_test_unscaled_mlp.flatten()
    raw_preds_flat_mlp = raw_preds_unscaled_mlp.flatten()

    # Calculate Regression Metrics
    r2_mlp = r2_score(y_test_flat_mlp, raw_preds_flat_mlp)
    mse_mlp = mean_squared_error(y_test_flat_mlp, raw_preds_flat_mlp)
    mae_mlp = mean_absolute_error(y_test_flat_mlp, raw_preds_flat_mlp)

    print(f"\n--- Simple MLP Regression Results ({target_protein}) ---")
    print(f"R2 Score: {r2_mlp:.4f}")
    print(f"Mean Squared Error (MSE): {mse_mlp:.4f}")
    print(f"Mean Absolute Error (MAE): {mae_mlp:.4f}")

    # Plot Regression results
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#F8F9FA')
    ax.set_facecolor('#F8F9FA')
    sns.regplot(x=y_test_flat_mlp, y=raw_preds_flat_mlp, scatter_kws={'color': '#03807F', 'alpha': 0.3},
                line_kws={'color': '#F7B80E', 'linewidth': 5})
    plt.title(f'MLP regression, {target_protein} (R$^2$: {r2_mlp:.2f}, MSE:{mse_mlp:.2})', fontsize=18)
    plt.xlabel(f'{target_protein} level', fontsize=16)
    plt.ylabel(f'Predicted {target_protein} level', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"./vib_mlp_regression_plot_{target_protein}.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    input_features = ['density', 'diversity', 'CD4', 'Cytokeratin', 'CD31', 'CD36', 'NKG2D', 'CDX2', 'ITLN1', 'CD68',
                      'CD34', 'CD117']
    target_protein = 'Ki67'
    df, dataset, in_features = create_dataset_wo_outliers("./23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv")
    mlp(device, dataset)
