import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool

class GNN_Predictor(nn.Module):
    """
    Graph Neural Network Predictor for materials property prediction using GINEConv.
    Implements the logic from MODEL_ARCHITECTURE_LOGIC.md.
    """
    def __init__(self, node_dim: int = 5, edge_dim: int = 6, hidden_dim: int = 64, num_layers: int = 3):
        super(GNN_Predictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 1. Node Embedding
        self.node_emb = nn.Linear(node_dim, hidden_dim)
        
        # 2. Edge Embedding (MLP)
        self.edge_emb = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3. Message Passing Layers
        self.gine_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # GINEConv requires an NN for mapping node features
            nn_gine = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.gine_layers.append(GINEConv(nn=nn_gine, edge_dim=hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
        # 4. Prediction Head (MLP)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Expected input shapes:
        - x: [Batch * Num_Atoms, 5]
        - edge_index: [2, Batch * Num_Edges]
        - edge_attr: [Batch * Num_Edges, 6]
        - batch: [Batch * Num_Atoms]
        """
        
        # 1. Node Embedding
        x = self.node_emb(x) # Shape: [Batch * Num_Atoms, hidden_dim]
        assert x.shape[-1] == self.hidden_dim, "Node embedding output mismatch"
        
        # 2. Edge Embedding
        edge_attr = self.edge_emb(edge_attr) # Shape: [Batch * Num_Edges, hidden_dim]
        assert edge_attr.shape[-1] == self.hidden_dim, "Edge embedding output mismatch"
        
        # 3. Message Passing Loop
        for gin_layer, batch_norm in zip(self.gine_layers, self.batch_norms):
            x = gin_layer(x, edge_index, edge_attr=edge_attr)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
            
        # 4. Readout
        # Aggregate node features into graph-level features
        x = global_add_pool(x, batch) # Shape: [Batch, hidden_dim]
        
        # 5. Prediction Head
        out = self.prediction_head(x) # Shape: [Batch, 1]
        
        return out
