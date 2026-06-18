import torch
import torch.nn as nn
from typing import List

class Trainer:
    """
    Handles the training loop for the GNN_Predictor.
    """
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = self.criterion(out, batch.y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            
        return total_loss / len(dataloader.dataset)
        
    def train(self, train_loader, epochs: int) -> List[float]:
        losses = []
        for epoch in range(epochs):
            loss = self.train_epoch(train_loader)
            losses.append(loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        return losses
