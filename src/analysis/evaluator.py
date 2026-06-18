import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

from src.data_pipeline.dataset import PolymerDataset
from src.data_pipeline.dataloader import get_dataloaders
from src.models.gnn_predictor import GNN_Predictor

sns.set_theme(style="whitegrid", palette="muted")

def evaluate_model(model, test_loader, device):
    model.eval()
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_y_true.append(batch.y.cpu().detach().numpy())
            all_y_pred.append(pred.cpu().detach().numpy())
            
    y_true = np.concatenate(all_y_true).flatten()
    y_pred = np.concatenate(all_y_pred).flatten()
    
    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'evaluation'))
    os.makedirs(reports_dir, exist_ok=True)
    
    metrics_path = os.path.join(reports_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R2 Score: {r2:.4f}\n")
        
    print(f"Metrics saved to {metrics_path}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    
    # Task A: Parity Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', s=50)
    
    # Diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
    
    # Text box
    textstr = f'$R^2$ = {r2:.3f}\nMAE = {mae:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
            
    plt.xlabel('Actual log solubility', fontsize=12)
    plt.ylabel('Predicted log solubility', fontsize=12)
    plt.title('Parity Plot', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'parity_plot.png'), dpi=300)
    plt.close()
    print("Saved parity_plot.png")
    
    # Task B: Residual Plot
    error = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, error, alpha=0.6, edgecolors='w', s=50)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Actual log solubility', fontsize=12)
    plt.ylabel('Residual (Actual - Predicted)', fontsize=12)
    plt.title('Residual Plot', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'residual_plot.png'), dpi=300)
    plt.close()
    print("Saved residual_plot.png")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    csv_path = os.path.join(base_dir, 'data', 'raw', 'esol', 'raw', 'delaney-processed.csv')
    df = pd.read_csv(csv_path)
    
    data_dir = os.path.join(base_dir, 'data')
    dataset = PolymerDataset(
        root=data_dir, 
        dataframe=df, 
        smiles_col='smiles', 
        target_col='measured log solubility in mols per litre'
    )
    
    _, _, test_loader = get_dataloaders(dataset, batch_size=32)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_path = os.path.join(base_dir, 'saved_models', 'best_model.pth')
    model = GNN_Predictor(node_dim=5, edge_dim=6, hidden_dim=64, num_layers=3)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    
    evaluate_model(model, test_loader, device)
