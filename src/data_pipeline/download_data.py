import os
import torch
from torch_geometric.datasets import MoleculeNet
from pathlib import Path

def download_and_verify_dataset():
    """
    Mendownload dataset MoleculeNet (ESOL) langsung dari server PyTorch Geometric.
    Dataset ini digunakan untuk menguji arsitektur GNN sebelum menggunakan data polimer asli.
    """
    # Menentukan path sesuai PROJECT_STRUCTURE.md
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Memulai download dataset MoleculeNet (ESOL) dari server PyG...")
    print("   (Ini mungkin memakan waktu beberapa detik tergantung koneksi internet)")
    
    try:
        # PyG akan otomatis mendownload, memproses, dan menyimpan cache file .pt
        dataset = MoleculeNet(root=str(raw_data_dir), name="ESOL")
        
        print("\n✅ Dataset berhasil di-download dan diproses!")
        print(f"📊 Jumlah total molekul dalam dataset: {len(dataset)}")
        
        # Verifikasi Struktur Data (Sesuai DOMAIN_KNOWLEDGE.md)
        sample = dataset[0]
        print("\n🔍 Verifikasi Struktur Graph (Sample Index 0):")
        print(f"   - Node Features (x) shape: {sample.x.shape}  <- [Jumlah Atom, Jumlah Fitur Atom]")
        print(f"   - Edge Index shape: {sample.edge_index.shape} <- [2, Jumlah Ikatan * 2 (Undirected)]")
        print(f"   - Edge Attributes (edge_attr) shape: {getattr(sample, 'edge_attr', 'Tidak Ada').shape if hasattr(sample, 'edge_attr') else 'Tidak Ada'}")
        print(f"   - Target (y) shape: {sample.y.shape}          <- [1, Nilai Regresi]")
        
        print("\n🎉 Pipeline Data Siap! Anda bisa lanjut ke FASE 1: Featurization.")
        
    except Exception as e:
        print(f"\n❌ Terjadi kesalahan saat download: {e}")
        print("Pastikan Anda sudah menginstall torch_geometric dengan benar.")

if __name__ == "__main__":
    download_and_verify_dataset()