# Project Structure & Directory Rules

AI, gunakan struktur folder di bawah ini sebagai **Peta Mutlak**. 
Anda DILARANG membuat folder baru di luar struktur ini atau menaruh file di root folder (kecuali file konfigurasi seperti `requirements.txt` atau `.env`). Setiap kali Anda menulis kode baru, Anda WAJIB menempatkannya di direktori yang sesuai.

## 🌳 Directory Tree

```text
shin-material/
│
├── .cursorrules               # Aturan sistem global AI
├── .gitignore                 # Mengabaikan file cache Python, venv, dan data besar
├── requirements.txt           # Daftar dependencies (PyTorch, PyG, RDKit, Streamlit)
├── README.md                  # Dokumentasi utama proyek
│
├── 📁 .vscode/ atau 📁 .idea/   # Konfigurasi IDE (jangan di-commit ke GitHub)
│
├── 📁 data/                   # Penyimpanan Dataset
│   ├── raw/                   # Dataset mentah (CSV/JSON, misal: Polymer Genome)
│   ├── processed/             # Cache hasil konversi PyG (file .pt)
│   └── external/              # Data dari pihak ketiga (jika ada)
│
├── 📁 notebooks/              # Jupyter Notebooks untuk EDA (Exploratory Data Analysis)
│   ├── 01_data_exploration.ipynb
│   └── 02_model_experiment.ipynb
│
├── 📁 src/                    # SOURCE CODE UTAMA (Modular & Reusable)
│   ├── __init__.py
│   │
│   ├── 📁 chemistry/          # Logika RDKit & Pemrosesan Molekul
│   │   ├── __init__.py
│   │   ├── featurizer.py      # Konversi SMILES -> PyG Data Object
│   │   └── validators.py      # Validasi aturan kimia (valensi, sanitasi)
│   │
│   ├── 📁 data_pipeline/      # PyTorch Geometric Dataset & DataLoader
│   │   ├── __init__.py
│   │   ├── dataset.py         # Custom InMemoryDataset
│   │   └── dataloader.py      # Setup DataLoader & Splitting (Train/Val/Test)
│   │
│   ├── 📁 models/             # Arsitektur Neural Network
│   │   ├── __init__.py
│   │   ├── gnn_predictor.py   # Implementasi GINEConv (Forward Prediction)
│   │   └── generative.py      # (Fase Lanjutan) GraphVAE / GFlowNet
│   │
│   ├── 📁 training/           # Training Loop & Evaluasi
│   │   ├── __init__.py
│   │   ├── trainer.py         # Loop training, validasi, dan checkpointing
│   │   └── metrics.py         # Kalkulasi MAE, RMSE, R2 Score
│   │
│   └── 📁 utils/              # Helper Functions
│       ├── __init__.py
│       ├── logger.py          # Setup Logging (logging module)
│       └── visualizer.py      # Fungsi untuk plot loss & visualisasi 2D molekul
│
├── 📁 app/                    # DEPLOYMENT & UI
│   ├── main.py                # Entry point untuk Streamlit
│   ├── components/            # UI Components (sidebar, forms)
│   └── assets/                # CSS, images, logos
│
├── 📁 configs/                # Konfigurasi Hyperparameter (YAML/JSON)
│   └── default_config.yaml    # Learning rate, batch size, hidden dims
│
└── 📁 tests/                  # UNIT TESTS (Wajib untuk mencegah bug)
    ├── test_featurizer.py     # Test RDKit parsing & tensor shapes
    ├── test_dataset.py        # Test PyG DataLoader
    └── test_model.py          # Test Forward pass GNN