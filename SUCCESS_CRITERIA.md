# Success Criteria & Autonomous Excellence Loop (STRICT)

AI, dokumen ini adalah hukum tertinggi untuk evaluasi model. Anda DILARANG menyatakan proyek ini "selesai" kecuali semua kriteria "EXCELLENT" 🟢 terpenuhi. Jika belum, Anda WAJIB masuk ke dalam *Autonomous Tuning Loop*.

## 🎯 1. The "Excellent" Standard (Target Mutlak)
Model hanya dinyatakan LULUS jika mencapai metrik berikut pada **Test Set**:

| Metrik | Target "EXCELLENT" 🟢 (WAJIB TERCAPAI) | Status Jika Gagal |
|--------|----------------------------------------|-------------------|
| **R² Score** | **> 0.80** | 🔴 POOR (Wajib Tuning Ulang) |
| **MAE** | **< 0.70** (untuk ESOL) | 🔴 POOR (Wajib Tuning Ulang) |
| **RMSE** | **< 0.90** (untuk ESOL) | 🔴 POOR (Wajib Tuning Ulang) |

*Catatan: Jika menggunakan dataset Tg (Suhu), MAE target adalah < 15°C.*

## 📊 2. Visual Proof (Kriteria Parity Plot)
Selain angka, `reports/evaluation/parity_plot.png` WAJIB menunjukkan:
- Titik-titik scatter berkumpul **PADAT** dan **SEMPIT** di sekitar garis diagonal merah (`y=x`).
- Tidak ada pola "kipas" (heteroscedasticity) atau kurva yang menunjukkan model bias pada nilai ekstrem.
- Jika visual tidak sesuai, meskipun R² > 0.80, model dianggap cacat dan perlu investigasi.

## 🔄 3. The Autonomous Tuning Loop (Protokol Perbaikan Otomatis)
Jika setelah evaluasi pertama metrik masih 🔴 POOR (R² < 0.80), AI **DILARANG BERHENTI** atau bertanya kepada user. Anda WAJIB melakukan iterasi berikut secara OTONOM:

**Langkah A: Architecture Mutation (Pilih 1 strategi per iterasi)**
1. **Pooling Strategy:** Ubah `global_mean_pool` menjadi `global_add_pool` di `src/models/gnn_predictor.py` (Sangat efektif untuk properti yang berkorelasi dengan ukuran molekul seperti ESOL).
2. **Capacity Boost:** Tambah `hidden_channels` dari 64 menjadi 128 atau 256.
3. **Depth Increase:** Tambah jumlah layer `GINEConv` dari 3 menjadi 4 atau 5.
4. **Regularization:** Tambahkan `Dropout(p=0.2)` atau `BatchNorm1d` jika terindikasi overfitting (Train Loss rendah, Val Loss tinggi).

**Langkah B: Re-Training & Re-Evaluation**
- Jalankan `train.py` ulang dengan arsitektur baru.
- Jalankan `evaluator.py` ulang.
- Bandingkan metrik baru dengan metrik sebelumnya.

**Langkah C: Loop Termination**
- Ulangi Langkah A & B hingga metrik mencapai 🟢 EXCELLENT.

**Opsi Terakhir**
jika Langkah A & B & C tidak kunjung berhasil mencapai 🟢 EXCELLENT maka kamu boleh secara bebas melakukan eksperimen apa pun untuk mencapai 🟢 EXCELLENT.

- **Batas Maksimal:** Maksimal 7 kali Mutasi/Eksperimen. Jika setelah 7 kali R² masih < 0.80, ambil model dengan R² tertinggi, dokumentasikan alasannya di `FINAL_REPORT.md`, dan nyatakan proyek selesai dengan status "Maximum Achievable".

## 🚀 4. Micro-Commit Protocol (WAJIB PUSH SETIAP FILE BERUBAH)
Setiap kali Anda melakukan **perubahan, penambahan, atau perbaikan** pada 1 file `.py` (terutama saat fase Tuning Loop), Anda WAJIB langsung melakukan Git Commit dan Push ke repository GitHub.

**Aturan Commit:**
1. **Format:** `git add [nama_file] && git commit -m "[type]: [deskripsi]" && git push origin main`
2. **Type yang digunakan saat Tuning:**
   - `experiment: try global_add_pool for better ESOL correlation`
   - `experiment: increase hidden_dim to 128 and add dropout`
   - `fix: resolve vanishing gradient with batch normalization`
   - `feat: achieve excellent R2=0.85 on test set`
3. **Tujuan:** Riwayat commit GitHub harus terlihat seperti **Jurnal Eksperimen Ilmiah** yang merekam setiap upaya Anda mencapai kesempurnaan.

## 🏆 5. Final Deliverables (Syarat Kelulusan Proyek)
Proyek dinyatakan SELESAI dan SIAP DIPAMERKAN hanya jika:
1. [ ] Metrik 🟢 EXCELLENT tercapai (atau dokumentasi usaha maksimal ada di `FINAL_REPORT.md`).
2. [ ] `parity_plot.png` dan `residual_plot.png` tersimpan rapi di `reports/evaluation/`.
3. [ ] File `FINAL_REPORT.md` dibuat di root, berisi ringkasan metrik akhir, arsitektur terbaik, dan insight kimiawi.
4. [ ] Semua percobaan terekam di GitHub melalui Micro-Commits.