import sys
import os

# Add the src directory to the sys.path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chemistry.featurizer import smiles_to_graph

def test_featurizer():
    # 1. Test normal SMILES: Ethanol ('CCO')
    print("Testing valid SMILES ('CCO')...")
    data = smiles_to_graph('CCO', target=1.5)
    
    assert data is not None, "Data should not be None for valid SMILES"
    assert data.x.shape[1] == 5, f"Expected node feature dimension 5, got {data.x.shape[1]}"
    assert data.edge_attr.shape[1] == 6, f"Expected edge feature dimension 6, got {data.edge_attr.shape[1]}"
    assert data.edge_index.shape[0] == 2, f"Expected edge index shape[0] to be 2, got {data.edge_index.shape[0]}"
    print("Valid SMILES test passed!")
    
    # Print shapes for verification
    print(f"Node Features (x) Shape: {list(data.x.shape)}")
    print(f"Edge Index Shape: {list(data.edge_index.shape)}")
    print(f"Edge Features (edge_attr) Shape: {list(data.edge_attr.shape)}")
    
    # 2. Test broken SMILES
    print("\nTesting broken SMILES ('CCO>>')...")
    broken_data = smiles_to_graph('CCO>>')
    assert broken_data is None, "Data should be None for invalid SMILES"
    print("Broken SMILES test passed!")

if __name__ == "__main__":
    test_featurizer()
    print("\nAll tests passed successfully.")
