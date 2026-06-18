import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from src.data_pipeline.dataset import PolymerDataset

def get_dataloaders(root_dir: str = 'data', batch_size: int = 32):
    """
    Menginisialisasi PolymerDataset dan mengembalikan DataLoader untuk
    Train (80%), Validation (10%), dan Test (10%).
    """
    dataset = PolymerDataset(root=root_dir)
    
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset menggunakan random_split
    generator = torch.Generator().manual_seed(42) # Reproducibility
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # Inisialisasi DataLoader
    # Train dengan shuffle=True, Val/Test dengan shuffle=False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
