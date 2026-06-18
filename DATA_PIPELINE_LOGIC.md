# Data Pipeline Algorithmic Logic (STRICT)

AI, saat menulis kode untuk memproses SMILES menjadi PyTorch Geometric (PyG) Graph, WAJIB mengikuti algoritma ini. Jangan membuat asumsi sendiri.

## 1. Pra-Pemrosesan RDKit
1. `mol = Chem.MolFromSmiles(smiles)`
2. IF `mol` is None -> `return None`
3. `mol = Chem.RemoveHs(mol)` (Hapus hidrogen implisit).
4. `Chem.SanitizeMol(mol)` -> Tangkap `SanitizeException`.

## 2. Node Features (Matriks X)
Buat matriks fitur node dengan dimensi `[Num_Atoms, 5]`. Ekstrak fitur ini untuk SETIAP atom:
1. Atomic Number (Integer, normalized) -> Dimensi 1
2. Degree (Integer, 0-5) -> Dimensi 1
3. Formal Charge (Integer) -> Dimensi 1
4. Num Implicit Hs (Integer) -> Dimensi 1
5. Aromaticity (Boolean, 0 or 1) -> Dimensi 1

## 3. Edge Features (Matriks E) & Adjacency
- Inisialisasi list kosong: `edge_indices = []`, `edge_attrs = []`
- Loop setiap `bond` di `mol.GetBonds()`:
  - `i = bond.GetBeginAtomIdx()`, `j = bond.GetEndAtomIdx()`
  - **WAJIB:** Tambahkan `[i, j]` DAN `[j, i]` ke `edge_indices` (Undirected).
  - Ekstrak fitur ikatan: Bond Type (One-Hot, 4: Single, Double, Triple, Aromatic), IsConjugated (1), IsInRing (1). Total = 6 Dimensi.
  - Tambahkan vektor fitur ini ke `edge_attrs` (Dua kali, untuk `[i,j]` dan `[j,i]`).
- Konversi ke tensor: `edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()`
- **ASSERTION WAJIB:** `assert edge_index.shape[0] == 2`