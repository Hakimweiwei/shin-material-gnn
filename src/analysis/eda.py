import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw

# Set up matplotlib style
sns.set_theme(style="whitegrid", palette="muted")

def run_eda(csv_path: str):
    print(f"Reading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Target column is "measured log solubility in mols per litre"
    target_col = "measured log solubility in mols per litre"
    smiles_col = "smiles"
    
    y = df[target_col].values
    smiles_list = df[smiles_col].values
    
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'eda'))
    os.makedirs(reports_dir, exist_ok=True)
    
    # Task A: Distribusi Target
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram + KDE
    sns.histplot(y, kde=True, ax=axes[0], color='blue')
    axes[0].set_title('Target Distribution (Histogram + KDE)', fontsize=14)
    axes[0].set_xlabel('Measured log solubility', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    
    # Boxplot
    sns.boxplot(y=y, ax=axes[1], color='orange')
    axes[1].set_title('Target Boxplot', fontsize=14)
    axes[1].set_ylabel('Measured log solubility', fontsize=12)
    
    plt.tight_layout()
    target_dist_path = os.path.join(reports_dir, 'target_distribution.png')
    plt.savefig(target_dist_path, dpi=300)
    plt.close()
    
    # Task B: Analisis Kompleksitas Graf (Molekul)
    num_nodes = []
    num_edges = []
    
    valid_mols = []
    valid_mols_count = 0
    for s in smiles_list:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                mol = Chem.RemoveHs(mol)
                num_nodes.append(mol.GetNumAtoms())
                num_edges.append(mol.GetNumBonds())
                if valid_mols_count < 20:
                    valid_mols.append(mol)
                    valid_mols_count += 1
        except Exception:
            continue
            
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.histplot(num_nodes, kde=False, bins=20, ax=ax2, color='green')
    
    mean_nodes = np.mean(num_nodes)
    median_nodes = np.median(num_nodes)
    ax2.axvline(mean_nodes, color='red', linestyle='--', label=f'Mean: {mean_nodes:.2f}')
    ax2.axvline(median_nodes, color='blue', linestyle='-.', label=f'Median: {median_nodes:.2f}')
    
    ax2.set_title('Graph Complexity: Number of Atoms per Molecule', fontsize=14)
    ax2.set_xlabel('Number of Atoms', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.legend()
    
    plt.tight_layout()
    graph_comp_path = os.path.join(reports_dir, 'graph_complexity.png')
    plt.savefig(graph_comp_path, dpi=300)
    plt.close()
    
    # Task C: Visualisasi Struktur Kimia 2D (Grid)
    legends = [f"Mol {i+1}" for i in range(len(valid_mols))]
    img = Draw.MolsToGridImage(valid_mols, molsPerRow=5, subImgSize=(400, 300), legends=legends)
    
    mol_grid_path = os.path.join(reports_dir, 'molecule_grid.png')
    img.save(mol_grid_path)
    
    print("Files successfully generated:")
    print(f"- {target_dist_path}")
    print(f"- {graph_comp_path}")
    print(f"- {mol_grid_path}")

if __name__ == "__main__":
    # Get absolute path relative to this script
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    csv_path = os.path.join(base_dir, 'data', 'raw', 'esol', 'raw', 'delaney-processed.csv')
    run_eda(csv_path)
