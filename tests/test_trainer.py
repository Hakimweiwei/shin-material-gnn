import sys
import os
import torch
import pandas as pd
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.gnn_predictor import GNN_Predictor
from src.data_pipeline.dataset import PolymerDataset
from src.data_pipeline.dataloader import get_dataloaders
from src.training.trainer import Trainer

def test_trainer():
    print("Testing Trainer loop...")
    # 1. Setup Data
    # Repeating 'CCO' and 'CCC' multiple times to form a batch
    dummy_data = {
        'smiles': ['CCO', 'CCC'] * 10,
        'target': [1.0, 2.0] * 10
    }
    df = pd.DataFrame(dummy_data)
    test_root = './tests/temp_trainer_data'
    
    if os.path.exists(test_root):
        shutil.rmtree(test_root)
        
    dataset = PolymerDataset(root=test_root, dataframe=df)
    train_loader, _, _ = get_dataloaders(dataset, batch_size=4, train_ratio=1.0, val_ratio=0.0)
    
    # 2. Setup Model, Optimizer, Loss
    model = GNN_Predictor(node_dim=5, edge_dim=6, hidden_dim=32, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    # 3. Setup Trainer
    trainer = Trainer(model, optimizer, criterion)
    
    # 4. Train
    epochs = 20
    losses = trainer.train(train_loader, epochs=epochs)
    
    print(f"Initial loss: {losses[0]:.4f}, Final loss: {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "Loss did not decrease during training!"
    print("Trainer test passed successfully!")
    
    if os.path.exists(test_root):
        shutil.rmtree(test_root)

if __name__ == "__main__":
    test_trainer()
