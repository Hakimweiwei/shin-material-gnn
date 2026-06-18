import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from typing import Tuple

def get_dataloaders(dataset, batch_size: int = 32, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Splits the PyG dataset into train, validation, and test sets and returns DataLoaders.
    Default batch_size is 32 as requested.
    """
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Use fixed generator for reproducible splits
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
