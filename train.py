import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from src.data_pipeline.dataset import PolymerDataset
from src.data_pipeline.dataloader import get_dataloaders
from src.models.gnn_predictor import GNN_Predictor
from src.training.trainer import Trainer

def run_training():
    print("Starting FASE 3: Model Training")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'data', 'raw', 'esol', 'raw', 'delaney-processed.csv')
    df = pd.read_csv(csv_path)
    
    # Process dataset
    data_dir = os.path.join(base_dir, 'data')
    dataset = PolymerDataset(
        root=data_dir, 
        dataframe=df, 
        smiles_col='smiles', 
        target_col='measured log solubility in mols per litre'
    )
    
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(dataset, batch_size=32)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = GNN_Predictor(node_dim=5, edge_dim=6, hidden_dim=64, num_layers=3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    trainer = Trainer(model, optimizer, criterion, device=device)
    
    # Target 1-2 epoch for testing
    epochs = 2
    losses = trainer.train(train_loader, epochs=epochs)
    
    # Save the model
    save_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    torch.save(model.state_dict(), best_model_path)
    print(f"Model saved to {best_model_path}")
    print("FASE 3 SUCCESS.")

if __name__ == "__main__":
    run_training()
