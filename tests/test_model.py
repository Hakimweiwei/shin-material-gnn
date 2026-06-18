import sys
import os
import torch
import pandas as pd
import shutil

# Add the src directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.gnn_predictor import GNN_Predictor
from src.data_pipeline.dataset import PolymerDataset
from src.data_pipeline.dataloader import get_dataloaders

def test_model():
    print("Testing GNN_Predictor model...")
    # 1. Setup Data
    dummy_data = {
        'smiles': ['CCO', 'CCC', 'CCN', 'c1ccccc1'],
        'target': [1.5, 2.0, 1.2, 0.5]
    }
    df = pd.DataFrame(dummy_data)
    test_root = './tests/temp_model_data'
    
    if os.path.exists(test_root):
        shutil.rmtree(test_root)
        
    dataset = PolymerDataset(root=test_root, dataframe=df)
    train_loader, _, _ = get_dataloaders(dataset, batch_size=2, train_ratio=1.0, val_ratio=0.0)
    
    # 2. Setup Model
    model = GNN_Predictor(node_dim=5, edge_dim=6, hidden_dim=64, num_layers=3)
    
    # 3. Test Forward Pass
    model.eval()
    batch = next(iter(train_loader))
    
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    
    print(f"Output shape: {list(out.shape)}")
    assert out.shape == (2, 1), f"Expected output shape (2, 1) because batch_size is 2, got {out.shape}"
    
    print("GNN_Predictor test passed successfully!")
    
    if os.path.exists(test_root):
        shutil.rmtree(test_root)

if __name__ == "__main__":
    test_model()
