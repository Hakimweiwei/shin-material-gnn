# GNN Model Architecture Logic (STRICT)

AI, gunakan `torch_geometric.nn.GINEConv`. Ikuti pseudocode forward pass ini:

## Forward Pass Steps:
1. **Node Embedding:** `x = self.node_emb(x)` -> Linear(in=5, out=hidden).
2. **Edge Embedding:** `edge_attr = self.edge_emb(edge_attr)` -> MLP(in=6, out=hidden).
3. **Message Passing Loop:** 
   ```python
   for gin_layer, batch_norm in zip(self.gine_layers, self.batch_norms):
       x = gin_layer(x, edge_index, edge_attr)
       x = batch_norm(x)
       x = F.relu(x)
       x = F.dropout(x, p=0.1, training=self.training)