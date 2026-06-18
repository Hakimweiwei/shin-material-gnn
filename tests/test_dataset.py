import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline.dataloader import get_dataloaders

def run_tests():
    print("--- Memulai PyG DataLoader Batch Verification ---")
    
    # Kita menggunakan default root='data'
    # Jika dataset belum diproses, proses ini akan memakan waktu
    train_loader, val_loader, test_loader = get_dataloaders(root_dir='data', batch_size=32)
    
    print(f"Total Batches in Train Loader: {len(train_loader)}")
    
    # Ambil 1 batch dari train_loader
    batch = next(iter(train_loader))
    
    print("\n[Shape Information]")
    print(f"Batch x shape (Total_Semua_Atom, 5): {batch.x.shape}")
    print(f"Batch edge_index shape (2, Total_Semua_Edge): {batch.edge_index.shape}")
    print(f"Batch edge_attr shape (Total_Semua_Edge, 6): {batch.edge_attr.shape}")
    print(f"Batch y shape (Num_Graphs, 1): {batch.y.shape}")
    
    if hasattr(batch, 'batch'):
        print(f"Batch batch tensor shape (Total_Semua_Atom): {batch.batch.shape}")
    
    # Validasi sesuai DATA_PIPELINE_LOGIC.md
    # Node feature: 5 dimensi, Edge feature: 6 dimensi
    assert batch.x.shape[1] == 5, f"Node features seharusnya memiliki 5 kolom, tapi didapatkan {batch.x.shape[1]}"
    assert batch.edge_attr.shape[1] == 6, f"Edge features seharusnya memiliki 6 kolom, tapi didapatkan {batch.edge_attr.shape[1]}"
    
    print("\n✅ DataLoader Test Passed!")

if __name__ == "__main__":
    run_tests()
