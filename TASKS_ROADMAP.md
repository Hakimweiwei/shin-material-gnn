# Tasks Roadmap & Execution Protocol

AI, Anda DILARANG mengerjakan semua fase sekaligus. Kerjakan per fase, berikan kodenya, lalu **BERHENTI dan TUNGGU PERSETUJUAN (Approval)** dari saya.

## FASE 1: Data Pipeline & Featurization (Current Phase)
- [ ] Setup struktur folder proyek (`src/chemistry/`, `src/models/`, dll).
- [ ] Buat modul `src/chemistry/featurizer.py` menggunakan RDKit.
- [ ] Implementasikan fungsi `smiles_to_graph(smiles: str, target: float) -> torch_geometric.data.Data`.
- [ ] Pastikan dimensi Node (5) dan Edge (6) sesuai `DATA_PIPELINE_LOGIC.md`.
- [ ] Buat script test untuk memvalidasi 1 SMILES normal ('CCO') dan 1 SMILES rusak.

## FASE 2: PyG Dataset & DataLoader
- [ ] Buat kelas `PolymerDataset` (inherit `InMemoryDataset`).
- [ ] Setup `DataLoader` dengan `batch_size=32`.

## FASE 3: Model Training
- [ ] Implementasikan `GNN_Predictor` menggunakan `GINEConv` sesuai `MODEL_ARCHITECTURE_LOGIC.md`.
- [ ] Buat training loop dan pastikan Loss menurun pada dummy data.

## FASE 4: Streamlit App
- [ ] Buat UI untuk input SMILES dan visualisasi 3D `py3Dmol`.