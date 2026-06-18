import sys
import os
import shutil
import pandas as pd
import torch

# Add the src directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_pipeline.dataset import PolymerDataset
from src.data_pipeline.dataloader import get_dataloaders

def test_dataloader():
    print("Setting up dummy data for DataLoader...")
    # Create 100 valid smiles ('CCO' has 3 atoms and 2 bonds (4 undirected edges))
    dummy_data = {
        'smiles': ['CCO'] * 100,
        'target': [1.0] * 100
    }
    df = pd.DataFrame(dummy_data)
    
    test_root = './tests/temp_data_loader'
    
    if os.path.exists(test_root):
        shutil.rmtree(test_root)
        
    print("Creating dataset...")
    dataset = PolymerDataset(root=test_root, dataframe=df)
    
    # Call get_dataloaders with batch_size=32
    print("Setting up DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(dataset, batch_size=32)
    
    print(f"Train size: {len(train_loader.dataset)}, Val size: {len(val_loader.dataset)}, Test size: {len(test_loader.dataset)}")
    
    assert len(train_loader.dataset) == 80, f"Expected train size 80, got {len(train_loader.dataset)}"
    assert len(val_loader.dataset) == 10, f"Expected val size 10, got {len(val_loader.dataset)}"
    assert len(test_loader.dataset) == 10, f"Expected test size 10, got {len(test_loader.dataset)}"
    
    # Check batch shapes for the first batch
    batch = next(iter(train_loader))
    print(f"Batch node shape: {list(batch.x.shape)}")
    print(f"Batch edge_index shape: {list(batch.edge_index.shape)}")
    
    assert batch.x.shape[0] == 32 * 3, f"Expected 96 nodes (32 * 3 atoms), got {batch.x.shape[0]}"
    assert batch.edge_index.shape[0] == 2, f"Expected edge_index[0] == 2, got {batch.edge_index.shape[0]}"
    
    print("DataLoader test passed successfully!")
    
    if os.path.exists(test_root):
        shutil.rmtree(test_root)

if __name__ == "__main__":
    test_dataloader()
