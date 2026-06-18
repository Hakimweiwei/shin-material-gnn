import torch
from torch_geometric.data import Data
from rdkit import Chem
from typing import Optional

def smiles_to_graph(smiles: str, target: float = 0.0) -> Optional[Data]:
    """
    Mengonversi representasi SMILES menjadi graf PyTorch Geometric (PyG).
    
    Args:
        smiles (str): Representasi string SMILES dari molekul.
        target (float): Nilai target Tg (Glass Transition Temperature).
        
    Returns:
        Optional[Data]: Objek PyG Data berisi fitur graf, atau None jika validasi gagal.
    """
    # 1. Pra-Pemrosesan RDKit
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        # Tangkap Exception apa pun dari RDKit (termasuk SanitizeException)
        return None
        
    # 2. Node Features (Matriks X)
    # Target dimensi: [Num_Atoms, 5]
    node_features = []
    for atom in mol.GetAtoms():
        # Fitur 1: Atomic Number (Integer, normalized) - Normalisasi sederhana dengan max elemen umum (contoh: 118)
        atomic_num = float(atom.GetAtomicNum()) / 118.0
        # Fitur 2: Degree (Integer, 0-5)
        degree = float(atom.GetDegree())
        # Fitur 3: Formal Charge (Integer)
        formal_charge = float(atom.GetFormalCharge())
        # Fitur 4: Num Implicit Hs (Integer)
        num_implicit_hs = float(atom.GetNumImplicitHs())
        # Fitur 5: Aromaticity (Boolean, 0 or 1)
        is_aromatic = 1.0 if atom.GetIsAromatic() else 0.0
        
        node_features.append([atomic_num, degree, formal_charge, num_implicit_hs, is_aromatic])
        
    # Shape: [Num_Atoms, 5]
    x = torch.tensor(node_features, dtype=torch.float)
    if x.shape[0] > 0:
        assert x.shape[1] == 5, f"Dimensi Node Feature salah! Diharapkan 5, didapat {x.shape[1]}"
        
    # 3. Edge Features (Matriks E) & Adjacency
    edge_indices = []
    edge_attrs = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Ekstrak fitur ikatan: Bond Type (One-Hot, 4: Single, Double, Triple, Aromatic)
        bond_type = bond.GetBondType()
        bt_features = [0.0, 0.0, 0.0, 0.0]
        
        if bond_type == Chem.rdchem.BondType.SINGLE:
            bt_features[0] = 1.0
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            bt_features[1] = 1.0
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            bt_features[2] = 1.0
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            bt_features[3] = 1.0
            
        # IsConjugated (1)
        is_conjugated = 1.0 if bond.GetIsConjugated() else 0.0
        # IsInRing (1)
        is_in_ring = 1.0 if bond.IsInRing() else 0.0
        
        # Total = 6 Dimensi
        attr = bt_features + [is_conjugated, is_in_ring]
        
        # WAJIB: Tambahkan [i, j] DAN [j, i] (Undirected Graph)
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        
        # Tambahkan vektor fitur ini 2x untuk pasangan [i,j] dan [j,i]
        edge_attrs.append(attr)
        edge_attrs.append(attr)
        
    if len(edge_indices) > 0:
        # Shape: [2, Num_Edges * 2]
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        # Shape: [Num_Edges * 2, 6]
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        # ASSERTION WAJIB
        assert edge_index.shape[0] == 2, f"Shape edge_index[0] harus 2, didapat {edge_index.shape[0]}"
        assert edge_attr.shape[1] == 6, f"Dimensi Edge Feature salah! Diharapkan 6, didapat {edge_attr.shape[1]}"
    else:
        # Handle edge case untuk molekul dengan 1 atom (tanpa ikatan kimia)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)
        
    # Shape: [1]
    y = torch.tensor([target], dtype=torch.float)
    
    # Bungkus menjadi object PyG Data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    return data

if __name__ == '__main__':
    # Test Cepat
    print("Mengeksekusi test cepat untuk featurizer.py...")
    
    # 1. SMILES Valid (Ethanol)
    smiles_valid = "CCO"
    data_valid = smiles_to_graph(smiles_valid, target=100.5)
    print(f"\n[Test Valid SMILES: '{smiles_valid}']")
    if data_valid is not None:
        print("Berhasil!")
        print(f"- Node Feature Matrix (x) Shape: {data_valid.x.shape}")
        print(f"- Edge Index Shape: {data_valid.edge_index.shape}")
        print(f"- Edge Attr Shape: {data_valid.edge_attr.shape}")
        print(f"- Target (y) Shape: {data_valid.y.shape}")
    else:
        print("Gagal: Mengembalikan None.")
        
    # 2. SMILES Rusak
    smiles_invalid = "xyz"
    data_invalid = smiles_to_graph(smiles_invalid)
    print(f"\n[Test Invalid SMILES: '{smiles_invalid}']")
    if data_invalid is None:
        print("Berhasil! (Program tidak crash dan mengembalikan None dengan aman).")
    else:
        print("Gagal: Diharapkan mengembalikan None, tetapi memberikan hasil.")
