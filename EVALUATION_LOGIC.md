# Evaluation & Parity Plot Logic (STRICT)

AI, Anda WAJIB membuktikan performa model secara matematis dan visual pada Test Set. Buat file `src/analysis/evaluator.py`.

## 1. Inference Pipeline (Pengumpulan Data Prediksi)
- Load model terbaik dari `models/best_model.pth`.
- Set `model.eval()` dan gunakan `with torch.no_grad():`.
- Loop melalui `test_loader`:
  - Pindahkan data ke `device`.
  - Prediksi: `pred = model(data.x, data.edge_index, data.edge_attr, data.batch)`.
  - **PENTING:** Kumpulkan `data.y` dan `pred` ke dalam list python.
- Setelah loop selesai, gabungkan list dan detach dari GPU menjadi array NumPy: 
  `y_true = np.concatenate(...).cpu().numpy()`
  `y_pred = np.concatenate(...).cpu().numpy()`

## 2. Kalkulasi Metrik (Scikit-Learn)
WAJIB menghitung 3 metrik ini menggunakan `sklearn.metrics`, print ke terminal, dan save ke `reports/evaluation/metrics.txt`:
1. **MAE** (Mean Absolute Error)
2. **RMSE** (Root Mean Squared Error)
3. **R2 Score** (Coefficient of Determination)

## 3. Task A: Parity Plot (Actual vs Predicted) - *SANGAT KRUSIAL*
Ini adalah grafik standar jurnal Materials Informatics.
- Buat Scatter Plot: Sumbu X = `y_true`, Sumbu Y = `y_pred`.
- Gunakan `alpha=0.6`, `edgecolors='w'`, `s=50` agar titik tidak bertumpuk (overplotting).
- **WAJIB:** Gambar garis diagonal merah putus-putus (`y = x`) dari nilai minimum hingga maksimum data. Garis ini merepresentasikan "Perfect Prediction".
- Tambahkan *Text Box* di pojok kiri atas plot berisi nilai $R^2$ dan MAE.
- Simpan sebagai `parity_plot.png` (dpi=300).

## 4. Task B: Residual Plot (Error Analysis)
- Hitung residual: `error = y_true - y_pred`.
- Buat Scatter Plot: Sumbu X = `y_true`, Sumbu Y = `error`.
- Tambahkan garis horizontal hitam di `y=0`.
- Plot ini untuk membuktikan model tidak bias (tidak selalu over-predict atau under-predict pada rentang suhu/kelarutan tertentu).
- Simpan sebagai `residual_plot.png`.

## 5. Aturan Eksekusi
- Pastikan semua tensor di-detach dari GPU sebelum di-plot (`tensor.cpu().detach().numpy()`).
- Bungkus dalam fungsi `evaluate_model(model, test_loader, device)`.