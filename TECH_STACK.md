# Tech Stack & Architecture Constraints

## 1. Core Environment
- **Language:** Python 3.10+ (Wajib gunakan Type Hinting `typing` di semua fungsi).
- **Chemistry:** `rdkit` (Standar industri).
- **Machine Learning:** `torch`, `torch_geometric` (PyG).
- **Data Handling:** `pandas`, `numpy`, `pydantic`.

## 2. Arsitektur Graph Neural Network (Strict Rules)
- **DILARANG** menggunakan `GCNConv` atau `GATConv` standar. Layer tersebut mengabaikan fitur ikatan kimia (edge attributes).
- **WAJIB** menggunakan `GINEConv` (Graph Isomorphism Network with Edge features).
- **Flow Arsitektur:**
  1. Node Embedding (Linear)
  2. Edge Embedding (MLP)
  3. Message Passing (3-4 layer GINEConv + BatchNorm + ReLU)
  4. Readout (`global_mean_pool`)
  5. Prediction Head (MLP -> Output 1 nilai float).

## 3. UI / Deployment
- **Framework:** `streamlit`
- **Visualisasi 3D:** `py3Dmol` (untuk merender molekul di browser).