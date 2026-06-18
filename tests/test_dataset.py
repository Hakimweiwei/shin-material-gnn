import sys
import os
import shutil
import pandas as pd
import torch

# Add the src directory to the sys.path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_pipeline.dataset import PolymerDataset

def test_dataset():
    print("Setting up dummy data...")
    dummy_data = {
        'smiles': ['CCO', 'CCC', 'CCN', 'INVALID_SMILES_SHOULD_BE_SKIPPED'],
        'target': [1.5, 2.0, 1.2, 0.0]
    }
    df = pd.DataFrame(dummy_data)
    
    test_root = './tests/temp_data'
    
    # Ensure clean state
    if os.path.exists(test_root):
        shutil.rmtree(test_root)
        
    print("Creating PolymerDataset...")
    dataset = PolymerDataset(root=test_root, dataframe=df, smiles_col='smiles', target_col='target')
    
    print(f"Dataset created with {len(dataset)} valid graphs.")
    
    # There are 3 valid smiles, 1 invalid, so length should be 3
    assert len(dataset) == 3, f"Expected dataset length 3, got {len(dataset)}"
    
    # Test a single data point
    data = dataset[0]
    assert data.x is not None, "Node features should exist"
    assert data.edge_index is not None, "Edge indices should exist"
    assert data.edge_attr is not None, "Edge attributes should exist"
    assert data.y is not None, "Target should exist"
    
    print("Dataset tests passed successfully!")
    
    # Cleanup
    if os.path.exists(test_root):
        shutil.rmtree(test_root)

if __name__ == "__main__":
    test_dataset()
