import os
import shutil
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from typing import Optional, Callable, List

# Import featurizer yang sudah dibuat pada FASE 1
from src.chemistry.featurizer import smiles_to_graph

class PolymerDataset(InMemoryDataset):
    """
    Custom PyG Dataset untuk memuat data polimer/molekul dari CSV.
    Akan mengeksekusi featurizer pada setiap string SMILES dan menyimpannya 
    ke file binary (.pt) agar load lebih cepat.
    """
    def __init__(self, root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        # Memuat data yang sudah diproses dari file .pt
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> List[str]:
        return ['delaney-processed.csv']
        
    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']
        
    def download(self):
        # Jika file CSV tidak ada di data/raw, periksa apakah file tersebut 
        # ada di direktori esol (bekas unduhan) dan salin jika ada.
        target_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        if not os.path.exists(target_path):
            esol_path = os.path.join('data', 'raw', 'esol', 'raw', 'delaney-processed.csv')
            if os.path.exists(esol_path):
                print(f"Menyalin dataset dari {esol_path} ke {target_path}...")
                shutil.copy2(esol_path, target_path)
            else:
                print(f"File raw tidak ditemukan di {target_path} dan tidak ada fallback. Pastikan file ada.")

    def process(self):
        raw_path = self.raw_paths[0]
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"File raw tidak ditemukan: {raw_path}")
            
        print("Membaca file CSV...")
        df = pd.read_csv(raw_path)
        
        # Mencari kolom SMILES dan target secara dinamis
        smiles_col = "smiles"
        target_col = "measured log solubility in mols per litre"
        
        # Fallback jika nama kolom sedikit berbeda
        if smiles_col not in df.columns:
            for c in df.columns:
                if 'SMILE' in c.upper():
                    smiles_col = c
                    break
        if target_col not in df.columns:
            for c in df.columns:
                if 'MEASURED' in c.upper() or 'SOLUBILITY' in c.upper() or 'TG' in c.upper():
                    target_col = c
                    break
                    
        print(f"Menggunakan kolom SMILES: '{smiles_col}', Target: '{target_col}'")
        
        data_list = []
        success_count = 0
        fail_count = 0
        
        print("Memulai konversi SMILES ke Graph...")
        for idx, row in df.iterrows():
            smiles = str(row[smiles_col])
            target = float(row[target_col])
            
            # Memanggil fungsi featurizer dari FASE 1
            data = smiles_to_graph(smiles, target)
            if data is not None:
                data_list.append(data)
                success_count += 1
            else:
                fail_count += 1
                
        print(f"Konversi Selesai! Berhasil: {success_count}, Gagal: {fail_count}")
        
        # Terapkan filter dan transform opsional jika ada
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Simpan objek List[Data] ke file binary PyG (data.pt)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
