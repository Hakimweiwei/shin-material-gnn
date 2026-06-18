import torch
import torch_geometric
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem
import numpy as np
from typing import Optional

def smiles_to_graph(smiles: str, target: float = 0.0) -> Optional[Data]:
    """
    Converts a SMILES string to a PyTorch Geometric Data object.
    Follows STRICT rules from DATA_PIPELINE_LOGIC.md
    """
    try:
        # 1. RDKit Pre-processing
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        
        # 2. Node Features (Matrix X) - Expected Shape: [Num_Atoms, 5]
        node_features = []
        for atom in mol.GetAtoms():
            # 1. Atomic Number (normalized by 118) -> Dim 1
            atomic_num = atom.GetAtomicNum() / 118.0
            # 2. Degree (Integer, 0-5) -> Dim 1
            degree = float(atom.GetDegree())
            # 3. Formal Charge -> Dim 1
            formal_charge = float(atom.GetFormalCharge())
            # 4. Num Implicit Hs -> Dim 1
            num_implicit_hs = float(atom.GetNumImplicitHs())
            # 5. Aromaticity -> Dim 1
            is_aromatic = 1.0 if atom.GetIsAromatic() else 0.0
            
            node_features.append([atomic_num, degree, formal_charge, num_implicit_hs, is_aromatic])
            
        x = torch.tensor(node_features, dtype=torch.float)
        
        # 3. Edge Features (Matrix E) & Adjacency - Expected Shape edge_attr: [Num_Edges*2, 6]
        edge_indices = []
        edge_attrs = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Undirected: add both [i, j] and [j, i]
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            
            # Extract features (6 dimensions)
            bond_type = bond.GetBondType()
            is_single = 1.0 if bond_type == Chem.rdchem.BondType.SINGLE else 0.0
            is_double = 1.0 if bond_type == Chem.rdchem.BondType.DOUBLE else 0.0
            is_triple = 1.0 if bond_type == Chem.rdchem.BondType.TRIPLE else 0.0
            is_aromatic = 1.0 if bond_type == Chem.rdchem.BondType.AROMATIC else 0.0
            is_conjugated = 1.0 if bond.GetIsConjugated() else 0.0
            is_in_ring = 1.0 if bond.IsInRing() else 0.0
            
            edge_feature = [is_single, is_double, is_triple, is_aromatic, is_conjugated, is_in_ring]
            
            # Add twice for undirected
            edge_attrs.append(edge_feature)
            edge_attrs.append(edge_feature)
            
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            # Handle single atom molecules with 0 edges
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 6), dtype=torch.float)
            
        # Assertion WAJIB
        assert edge_index.shape[0] == 2, f"Expected edge_index.shape[0] == 2, got {edge_index.shape[0]}"
        
        # Target node/graph property (Shape: [1, 1])
        y = torch.tensor([[target]], dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data

    except Exception as e:
        # In case of SanitizeException or other errors
        return None
