# Domain Knowledge: Chemistry & Materials Informatics

AI, Anda sedang berurusan dengan kimia komputasional. Patuhi hukum fisika dan kimia berikut:

## 1. Representasi Graf Molekul
- **Node (Titik):** Merepresentasikan Atom. 
- **Edge (Garis):** Merepresentasikan Ikatan Kimia (Bonds).
- **Graf Tidak Berarah (Undirected):** Molekul adalah graf undirected. Setiap ikatan antara atom A dan B HARUS direpresentasikan sebagai dua edge: (A -> B) dan (B -> A).

## 2. Properti Material Target: Glass Transition Temperature (Tg)
- **Definisi:** Suhu di mana polimer berubah dari keras/rapuh (glassy) menjadi lunak/fleksibel (rubbery).
- **Relevansi Otomotif:** Casing baterai EV membutuhkan polimer dengan Tg tinggi agar tidak meleleh saat suhu operasional naik.
- **Tipe Prediksi:** Tg adalah properti *intensif* (tidak bergantung pada massa/ukuran absolut molekul). Oleh karena itu, hasil akhir GNN WAJIB di-pooling menggunakan `global_mean_pool`, BUKAN `global_add_pool`.

## 3. Aturan SMILES & RDKit
- **Valensi:** Karbon (C) maksimal 4 ikatan, Oksigen (O) maksimal 2. Jangan pernah memproses struktur yang melanggar aturan valensi.
- **Sanitasi:** Setiap molekul dari dataset HARUS melewati `Chem.SanitizeMol()`. Jika error, buang molekul tersebut (return None).