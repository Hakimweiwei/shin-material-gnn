# EDA (Exploratory Data Analysis) Logic (STRICT)

AI, sebelum melatih model, Anda WAJIB memahami distribusi data dan struktur molekul. Buat file `src/analysis/eda.py` dan ikuti spesifikasi visual berikut.

## 1. Environment & Setup Visual
- Library: `matplotlib.pyplot`, `seaborn`, `pandas`, `rdkit.Chem`, `rdkit.Chem.Draw`.
- Style: WAJIB gunakan `sns.set_theme(style="whitegrid", palette="muted")`.
- Font: Pastikan label sumbu dan judul terbaca jelas (fontsize=12 untuk label, 14 untuk title).
- Output: Semua gambar WAJIB disimpan di folder `reports/eda/` dengan resolusi `dpi=300`.

## 2. Task A: Distribusi Target (Water Solubility / ESOL)
- Ekstrak semua nilai `y` dari dataset PyG.
- Buat Figure dengan 2 Subplots berdampingan (1 baris, 2 kolom):
  1. **Histogram + KDE:** Plot distribusi nilai target dengan kurva KDE (Kernel Density Estimate) di atasnya.
  2. **Boxplot:** Untuk mendeteksi adanya outlier ekstrem pada dataset.
- Simpan sebagai `target_distribution.png`.

## 3. Task B: Analisis Kompleksitas Graf (Molekul)
- Loop melalui dataset, ekstrak `data.num_nodes` (jumlah atom) dan `data.num_edges` (jumlah ikatan).
- Buat Histogram distribusi jumlah atom per molekul.
- Tambahkan garis vertikal `mean` dan `median` dengan warna berbeda.
- Simpan sebagai `graph_complexity.png`.

## 4. Task C: Visualisasi Struktur Kimia 2D (Grid)
- Baca 20 string SMILES pertama dari file CSV mentah (`delaney-processed.csv`).
- Konversi ke objek RDKit Mol.
- Gunakan `Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(400, 300), legends=[...])`.
- Simpan sebagai `molecule_grid.png`.

## 5. Aturan Eksekusi
- Bungkus semua proses dalam fungsi `run_eda(dataset, csv_path)`.
- Gunakan `try-except` saat menggambar molekul. Jika 1 molekul gagal di-parse RDKit, skip dan lanjut ke molekul berikutnya (jangan crash).
- Di akhir script, print daftar file gambar yang berhasil di-generate.