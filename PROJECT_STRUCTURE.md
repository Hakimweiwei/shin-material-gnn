# Project Structure & Directory Rules

AI, gunakan struktur folder di bawah ini sebagai **Peta Mutlak**. 
Anda DILARANG membuat folder baru di luar struktur ini atau menaruh file di root folder (kecuali file konfigurasi seperti `requirements.txt` atau `.env`). Setiap kali Anda menulis kode baru, Anda WAJIB menempatkannya di direktori yang sesuai.

## 🌳 Directory Tree

```text
shin-material/
│
├── .antigravityrules          # Aturan sistem global AI (Wajib di root)
├── .gitignore                 # Mengabaikan venv, cache PyG, dan file besar
├── requirements.txt           # Daftar dependencies Python
├── README.md                  # Dokumentasi utama proyek
├── train.py                   # Entry point utama untuk menjalankan training
│
├── 📁 [Context Files]         # Kumpulan file .md (PROJECT_CONTEXT, DOMAIN_KNOWLEDGE, dll)
│
├── 📁 data/                   # Penyimpanan Dataset
│   ├── raw/                   # Dataset mentah (delaney-processed.csv / ESOL)
│   └── processed/             # Cache hasil konversi PyG (file .pt) -> ABAIKAN DI GIT
│
├── 📁 reports/                # OUTPUT VISUAL & LAPORAN (Wajib di-commit ke GitHub)
│   ├── eda/                   # Histogram, Grid Molekul (dari eda.py)
│   └── evaluation/            # Parity Plot, Residual Plot, metrics.txt (dari evaluator.py)
│
├── 📁 saved_models/           # Bobot Model Hasil Training (.pth)
│   └── best_model.pth         # Disimpan oleh trainer.py berdasarkan Val Loss terendah
│
├── 📁 src/                    # SOURCE CODE UTAMA (Modular & Reusable)
│   ├── __init__.py
│   │
│   ├── 📁 chemistry/          # Logika RDKit & Pemrosesan Molekul
│   │   ├── featurizer.py      # Konversi SMILES -> PyG Data Object (Node=5, Edge=6)
│   │   └── validators.py      # Validasi aturan kimia (valensi, sanitasi)
│   │
│   ├── 📁 data_pipeline/      # PyTorch Geometric Dataset & DataLoader
│   │   ├── dataset.py         # Custom InMemoryDataset
│   │   └── dataloader.py      # Setup DataLoader & Splitting (Train/Val/Test)
│   │
│   ├── 📁 models/             # Arsitektur Neural Network (CODE ONLY)
│   │   └── gnn_predictor.py   # Implementasi GINEConv (Forward Prediction)
│   │
│   ├── 📁 training/           # Training Loop & Evaluasi
│   │   ├── trainer.py         # Loop training, validasi, dan checkpointing
│   │   └── metrics.py         # Kalkulasi MAE, RMSE, R2 Score
│   │
│   ├── 📁 analysis/           # EDA & EVALUATION SCRIPTS
│   │   ├── eda.py             # Generate plot distribusi & grid molekul
│   │   └── evaluator.py       # Inference test set & generate Parity Plot
│   │
│   └── 📁 utils/              # Helper Functions
│       ├── logger.py          # Setup Logging
│       └── visualizer.py      # Fungsi bantuan untuk plot
│
├── 📁 configs/                # Konfigurasi Hyperparameter
│   └── default_config.yaml    # Learning rate, batch size, hidden dims
│
└── 📁 tests/                  # UNIT TESTS (Wajib untuk mencegah bug)
    ├── test_featurizer.py     # Test RDKit parsing & tensor shapes
    ├── test_dataset.py        # Test PyG DataLoader batching
    └── test_model.py          # Test Forward pass GNN